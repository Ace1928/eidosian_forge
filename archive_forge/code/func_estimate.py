import gymnasium as gym
import numpy as np
import tree
from typing import Dict, Any, List
import logging
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import convert_ma_batch_to_sample_batch
from ray.rllib.utils.policy import compute_log_likelihoods_from_input_dict
from ray.rllib.utils.annotations import (
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.typing import TensorType, SampleBatchType
from ray.rllib.offline.offline_evaluator import OfflineEvaluator
@DeveloperAPI
def estimate(self, batch: SampleBatchType, split_batch_by_episode: bool=True) -> Dict[str, Any]:
    """Compute off-policy estimates.

        Args:
            batch: The batch to calculate the off-policy estimates (OPE) on. The
            batch must contain the fields "obs", "actions", and "action_prob".
            split_batch_by_episode: Whether to split the batch by episode.

        Returns:
            The off-policy estimates (OPE) calculated on the given batch. The returned
            dict can be any arbitrary mapping of strings to metrics.
            The dict consists of the following metrics:
            - v_behavior: The discounted return averaged over episodes in the batch
            - v_behavior_std: The standard deviation corresponding to v_behavior
            - v_target: The estimated discounted return for `self.policy`,
            averaged over episodes in the batch
            - v_target_std: The standard deviation corresponding to v_target
            - v_gain: v_target / max(v_behavior, 1e-8)
            - v_delta: The difference between v_target and v_behavior.
        """
    batch = convert_ma_batch_to_sample_batch(batch)
    self.check_action_prob_in_batch(batch)
    estimates_per_epsiode = []
    if split_batch_by_episode:
        batch = self.on_before_split_batch_by_episode(batch)
        all_episodes = batch.split_by_episode()
        all_episodes = self.on_after_split_batch_by_episode(all_episodes)
        for episode in all_episodes:
            assert len(set(episode[SampleBatch.EPS_ID])) == 1, 'The episode must contain only one episode id. For some reason the split_by_episode() method could not successfully split the batch by episodes. Each row in the dataset should be one episode. Check your evaluation dataset for errors.'
            self.peek_on_single_episode(episode)
        for episode in all_episodes:
            estimate_step_results = self.estimate_on_single_episode(episode)
            estimates_per_epsiode.append(estimate_step_results)
        estimates_per_epsiode = tree.map_structure(lambda *x: list(x), *estimates_per_epsiode)
    else:
        estimates_per_epsiode = self.estimate_on_single_step_samples(batch)
    estimates = {'v_behavior': np.mean(estimates_per_epsiode['v_behavior']), 'v_behavior_std': np.std(estimates_per_epsiode['v_behavior']), 'v_target': np.mean(estimates_per_epsiode['v_target']), 'v_target_std': np.std(estimates_per_epsiode['v_target'])}
    estimates['v_gain'] = estimates['v_target'] / max(estimates['v_behavior'], 1e-08)
    estimates['v_delta'] = estimates['v_target'] - estimates['v_behavior']
    return estimates
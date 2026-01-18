import argparse
import numpy as np
from gymnasium.spaces import Discrete
import os
import ray
from ray import air, tune
from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo_tf_policy import (
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing
from ray.rllib.examples.env.two_step_game import TwoStepGame
from ray.rllib.examples.models.centralized_critic_models import (
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.tf_utils import explained_variance, make_tf_callable
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
def centralized_critic_postprocessing(policy, sample_batch, other_agent_batches=None, episode=None):
    pytorch = policy.config['framework'] == 'torch'
    if pytorch and hasattr(policy, 'compute_central_vf') or (not pytorch and policy.loss_initialized()):
        assert other_agent_batches is not None
        if policy.config['enable_connectors']:
            [(_, _, opponent_batch)] = list(other_agent_batches.values())
        else:
            [(_, opponent_batch)] = list(other_agent_batches.values())
        sample_batch[OPPONENT_OBS] = opponent_batch[SampleBatch.CUR_OBS]
        sample_batch[OPPONENT_ACTION] = opponent_batch[SampleBatch.ACTIONS]
        if args.framework == 'torch':
            sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(convert_to_torch_tensor(sample_batch[SampleBatch.CUR_OBS], policy.device), convert_to_torch_tensor(sample_batch[OPPONENT_OBS], policy.device), convert_to_torch_tensor(sample_batch[OPPONENT_ACTION], policy.device)).cpu().detach().numpy()
        else:
            sample_batch[SampleBatch.VF_PREDS] = convert_to_numpy(policy.compute_central_vf(sample_batch[SampleBatch.CUR_OBS], sample_batch[OPPONENT_OBS], sample_batch[OPPONENT_ACTION]))
    else:
        sample_batch[OPPONENT_OBS] = np.zeros_like(sample_batch[SampleBatch.CUR_OBS])
        sample_batch[OPPONENT_ACTION] = np.zeros_like(sample_batch[SampleBatch.ACTIONS])
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(sample_batch[SampleBatch.REWARDS], dtype=np.float32)
    completed = sample_batch[SampleBatch.TERMINATEDS][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]
    train_batch = compute_advantages(sample_batch, last_r, policy.config['gamma'], policy.config['lambda'], use_gae=policy.config['use_gae'])
    return train_batch
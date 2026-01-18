import numpy as np
import pandas as pd
from typing import Any, Dict, Type, TYPE_CHECKING
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy import Policy
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.annotations import DeveloperAPI
@DeveloperAPI
def compute_is_weights(batch: pd.DataFrame, policy_state: Dict[str, Any], estimator_class: Type['OffPolicyEstimator']) -> pd.DataFrame:
    """Computes the importance sampling weights for the given batch of samples.

    For a lot of off-policy estimators, the importance sampling weights are computed as
    the propensity score ratio between the new and old policies
    (i.e. new_pi(act|obs) / old_pi(act|obs)). This function is to be used with
    map_batches() to perform a batch prediction on a dataset of records with `obs`,
    `actions`, `action_prob` and `rewards` columns.

    Args:
        batch: A sub-batch from the dataset.
        policy_state: The state of the policy to use for the prediction.
        estimator_class: The estimator class to use for the prediction. This class

    Returns:
        The modified batch with the importance sampling weights, weighted rewards, new
        and old propensities added as columns.
    """
    policy = Policy.from_state(policy_state)
    estimator = estimator_class(policy=policy, gamma=0, epsilon_greedy=0)
    sample_batch = SampleBatch({SampleBatch.OBS: np.vstack(batch['obs'].values), SampleBatch.ACTIONS: np.vstack(batch['actions'].values).squeeze(-1), SampleBatch.ACTION_PROB: np.vstack(batch['action_prob'].values).squeeze(-1), SampleBatch.REWARDS: np.vstack(batch['rewards'].values).squeeze(-1)})
    new_prob = estimator.compute_action_probs(sample_batch)
    old_prob = sample_batch[SampleBatch.ACTION_PROB]
    rewards = sample_batch[SampleBatch.REWARDS]
    weights = new_prob / old_prob
    weighted_rewards = weights * rewards
    batch['weights'] = weights
    batch['weighted_rewards'] = weighted_rewards
    batch['new_prob'] = new_prob
    batch['old_prob'] = old_prob
    return batch
import numpy as np
from ray.rllib.algorithms.dreamerv3.utils.debugging import (
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.tf_utils import inverse_symlog
def report_predicted_vs_sampled_obs(*, results, sample, batch_size_B, batch_length_T, symlog_obs: bool=True):
    """Summarizes sampled data (from the replay buffer) vs world-model predictions.

    World model predictions are based on the posterior states (z computed from actual
    observation encoder input + the current h-states).

    Observations: Computes MSE (sampled vs predicted/recreated) over all features.
    For image observations, also creates direct image comparisons (sampled images
    vs predicted (posterior) ones).
    Rewards: Compute MSE (sampled vs predicted).
    Continues: Compute MSE (sampled vs predicted).

    Args:
        results: The results dict that was returned by `LearnerGroup.update()`.
        sample: The sampled data (dict) from the replay buffer. Already tf-tensor
            converted.
        batch_size_B: The batch size (B). This is the number of trajectories sampled
            from the buffer.
        batch_length_T: The batch length (T). This is the length of an individual
            trajectory sampled from the buffer.
    """
    predicted_observation_means_BxT = results['WORLD_MODEL_fwd_out_obs_distribution_means_BxT']
    _report_obs(results=results, computed_float_obs_B_T_dims=np.reshape(predicted_observation_means_BxT, (batch_size_B, batch_length_T) + sample[SampleBatch.OBS].shape[2:]), sampled_obs_B_T_dims=sample[SampleBatch.OBS], descr_prefix='WORLD_MODEL', descr_obs=f'predicted_posterior_T{batch_length_T}', symlog_obs=symlog_obs)
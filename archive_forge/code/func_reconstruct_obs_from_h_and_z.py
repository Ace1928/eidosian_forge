import numpy as np
from ray.rllib.algorithms.dreamerv3.utils.debugging import (
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.tf_utils import inverse_symlog
def reconstruct_obs_from_h_and_z(h_t0_to_H, z_t0_to_H, dreamer_model, obs_dims_shape):
    """Returns"""
    shape = h_t0_to_H.shape
    T = shape[0]
    B = shape[1]
    reconstructed_obs_distr_means_TxB = dreamer_model.world_model.decoder(h=np.reshape(h_t0_to_H, (T * B, -1)), z=np.reshape(z_t0_to_H, (T * B,) + z_t0_to_H.shape[2:]))
    reconstructed_obs_T_B = np.reshape(reconstructed_obs_distr_means_TxB, (T, B) + obs_dims_shape)
    return reconstructed_obs_T_B
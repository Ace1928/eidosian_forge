import numpy as np
from ray.rllib.algorithms.dreamerv3.utils.debugging import (
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.tf_utils import inverse_symlog
def report_dreamed_trajectory(*, results, env, dreamer_model, obs_dims_shape, batch_indices=(0,), desc=None, include_images=True):
    if not include_images:
        return
    dream_data = results['dream_data']
    dreamed_obs_H_B = reconstruct_obs_from_h_and_z(h_t0_to_H=dream_data['h_states_t0_to_H_BxT'], z_t0_to_H=dream_data['z_states_prior_t0_to_H_BxT'], dreamer_model=dreamer_model, obs_dims_shape=obs_dims_shape)
    func = create_cartpole_dream_image if env.startswith('CartPole') else create_frozenlake_dream_image
    for b in batch_indices:
        images = []
        for t in range(len(dreamed_obs_H_B) - 1):
            images.append(func(dreamed_obs=dreamed_obs_H_B[t][b], dreamed_V=dream_data['values_dreamed_t0_to_H_BxT'][t][b], dreamed_a=dream_data['actions_ints_dreamed_t0_to_H_BxT'][t][b], dreamed_r_tp1=dream_data['rewards_dreamed_t0_to_H_BxT'][t + 1][b], dreamed_ri_tp1=results['DISAGREE_intrinsic_rewards_H_BxT'][t][b] if 'DISAGREE_intrinsic_rewards_H_BxT' in results else None, dreamed_c_tp1=dream_data['continues_dreamed_t0_to_H_BxT'][t + 1][b], value_target=results['VALUE_TARGETS_H_BxT'][t][b], initial_h=dream_data['h_states_t0_to_H_BxT'][t][b], as_tensor=True).numpy())
        results.update({f'dreamed_trajectories{('_' + desc if desc else '')}_B{b}': np.concatenate(images, axis=1)})
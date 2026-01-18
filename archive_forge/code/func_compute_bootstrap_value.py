from typing import List
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.core.models.base import STATE_IN
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.evaluation.postprocessing import discount_cumsum
from ray.rllib.policy.sample_batch import concat_samples, SampleBatch
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.typing import TensorType
def compute_bootstrap_value(episode: SingleAgentEpisode, module: RLModule) -> SingleAgentEpisode:
    if episode.is_terminated:
        last_r = 0.0
    else:
        initial_states = module.get_initial_state()
        state = {k: initial_states[k] if episode.states is None else episode.states[k] for k in initial_states.keys()}
        input_dict = {STATE_IN: tree.map_structure(lambda s: convert_to_torch_tensor(s) if module.framework == 'torch' else tf.convert_to_tensor(s), state), SampleBatch.OBS: convert_to_torch_tensor(np.expand_dims(episode.observations[-1], axis=0)) if module.framework == 'torch' else tf.convert_to_tensor(np.expand_dims(episode.observations[-1], axis=0))}
        input_dict = NestedDict(input_dict)
        fwd_out = module.forward_exploration(input_dict)
        last_r = fwd_out[SampleBatch.VF_PREDS][-1]
    vf_preds = episode.extra_model_outputs[SampleBatch.VF_PREDS]
    episode.extra_model_outputs[SampleBatch.VALUES_BOOTSTRAPPED] = np.concatenate([vf_preds[1:], np.array([convert_to_numpy(last_r)], dtype=np.float32)], axis=0)
    return episode
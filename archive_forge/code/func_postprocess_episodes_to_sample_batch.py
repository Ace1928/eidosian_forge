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
@DeveloperAPI
def postprocess_episodes_to_sample_batch(episodes: List[SingleAgentEpisode]) -> SampleBatch:
    """Converts the results from sampling with an `EnvRunner` to one `SampleBatch'.

    Once the `SampleBatch` will be deprecated this function will be
    deprecated, too.
    """
    batches = []
    for episode_or_list in episodes:
        if isinstance(episode_or_list, list):
            for episode in episode_or_list:
                batches.append(episode.get_sample_batch())
        else:
            batches.append(episode_or_list.get_sample_batch())
    batch = concat_samples(batches)
    if SampleBatch.INFOS in batch.keys():
        del batch[SampleBatch.INFOS]
    return batch
import gymnasium as gym
import numpy as np
from typing import Optional, List, Mapping, Iterable, Dict
import tree
import abc
from ray.rllib.models.distributions import Distribution
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType, Union, Tuple
@classmethod
@override(Distribution)
def from_logits(cls, logits: torch.Tensor, child_distribution_cls_struct: Union[Mapping, Iterable], input_lens: Union[Dict, List[int]], space: gym.Space, **kwargs) -> 'TorchMultiDistribution':
    """Creates this Distribution from logits (and additional arguments).

        If you wish to create this distribution from logits only, please refer to
        `Distribution.get_partial_dist_cls()`.

        Args:
            logits: The tensor containing logits to be separated by `input_lens`.
                child_distribution_cls_struct: A struct of Distribution classes that can
                be instantiated from the given logits.
            child_distribution_cls_struct: A struct of Distribution classes that can
                be instantiated from the given logits.
            input_lens: A list or dict of integers that indicate the length of each
                logit. If this is given as a dict, the structure should match the
                structure of child_distribution_cls_struct.
            space: The possibly nested output space.
            **kwargs: Forward compatibility kwargs.

        Returns:
            A TorchMultiActionDistribution object.
        """
    logit_lens = tree.flatten(input_lens)
    child_distribution_cls_list = tree.flatten(child_distribution_cls_struct)
    split_logits = torch.split(logits, logit_lens, dim=1)
    child_distribution_list = tree.map_structure(lambda dist, input_: dist.from_logits(input_), child_distribution_cls_list, list(split_logits))
    child_distribution_struct = tree.unflatten_as(child_distribution_cls_struct, child_distribution_list)
    return TorchMultiDistribution(child_distribution_struct=child_distribution_struct)
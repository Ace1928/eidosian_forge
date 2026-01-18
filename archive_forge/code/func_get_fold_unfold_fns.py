import logging
import numpy as np
import tree  # pip install dm_tree
from typing import List, Optional
import functools
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.debug import summarize
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.typing import TensorType, ViewRequirementsDict
from ray.util import log_once
from ray.rllib.utils.typing import SampleBatchType
@DeveloperAPI
def get_fold_unfold_fns(b_dim: int, t_dim: int, framework: str):
    """Produces two functions to fold/unfold any Tensors in a struct.

    Args:
        b_dim: The batch dimension to use for folding.
        t_dim: The time dimension to use for folding.
        framework: The framework to use for folding. One of "tf2" or "torch".

    Returns:
        fold: A function that takes a struct of torch.Tensors and reshapes
            them to have a first dimension of `b_dim * t_dim`.
        unfold: A function that takes a struct of torch.Tensors and reshapes
            them to have a first dimension of `b_dim` and a second dimension
            of `t_dim`.
    """
    if framework in 'tf2':
        b_dim = tf.convert_to_tensor(b_dim)
        t_dim = tf.convert_to_tensor(t_dim)

        def fold_mapping(item):
            if item is None:
                return item
            item = tf.convert_to_tensor(item)
            shape = tf.shape(item)
            other_dims = shape[2:]
            return tf.reshape(item, tf.concat([[b_dim * t_dim], other_dims], axis=0))

        def unfold_mapping(item):
            if item is None:
                return item
            item = tf.convert_to_tensor(item)
            shape = item.shape
            other_dims = shape[1:]
            return tf.reshape(item, tf.concat([[b_dim], [t_dim], other_dims], axis=0))
    elif framework == 'torch':

        def fold_mapping(item):
            if item is None:
                return item
            item = torch.as_tensor(item)
            size = list(item.size())
            current_b_dim, current_t_dim = list(size[:2])
            assert (b_dim, t_dim) == (current_b_dim, current_t_dim), 'All tensors in the struct must have the same batch and time dimensions. Got {} and {}.'.format((b_dim, t_dim), (current_b_dim, current_t_dim))
            other_dims = size[2:]
            return item.reshape([b_dim * t_dim] + other_dims)

        def unfold_mapping(item):
            if item is None:
                return item
            item = torch.as_tensor(item)
            size = list(item.size())
            current_b_dim = size[0]
            other_dims = size[1:]
            assert current_b_dim == b_dim * t_dim, 'The first dimension of the tensor must be equal to the product of the desired batch and time dimensions. Got {} and {}.'.format(current_b_dim, b_dim * t_dim)
            return item.reshape([b_dim, t_dim] + other_dims)
    else:
        raise ValueError(f'framework {framework} not implemented!')
    return (functools.partial(tree.map_structure, fold_mapping), functools.partial(tree.map_structure, unfold_mapping))
from gymnasium.spaces import Discrete, Box, MultiDiscrete, Space
import numpy as np
import tree  # pip install dm_tree
from typing import Union, Optional
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils import force_tuple
from ray.rllib.utils.framework import try_import_tf, try_import_torch, TensorType
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.tf_utils import zero_logps_from_actions
def random_component(component):
    shape = component.shape or (1,)
    if isinstance(component, Discrete):
        return tf.random.uniform(shape=(batch_size,) + component.shape, maxval=component.n, dtype=component.dtype)
    elif isinstance(component, MultiDiscrete):
        return tf.concat([tf.random.uniform(shape=(batch_size, 1), maxval=n, dtype=component.dtype) for n in component.nvec], axis=1)
    elif isinstance(component, Box):
        if component.bounded_above.all() and component.bounded_below.all():
            if component.dtype.name.startswith('int'):
                return tf.random.uniform(shape=(batch_size,) + shape, minval=component.low.flat[0], maxval=component.high.flat[0], dtype=component.dtype)
            else:
                return tf.random.uniform(shape=(batch_size,) + shape, minval=component.low, maxval=component.high, dtype=component.dtype)
        else:
            return tf.random.normal(shape=(batch_size,) + shape, dtype=component.dtype)
    else:
        assert isinstance(component, Simplex), "Unsupported distribution component '{}' for random sampling!".format(component)
        return tf.nn.softmax(tf.random.uniform(shape=(batch_size,) + shape, minval=0.0, maxval=1.0, dtype=component.dtype))
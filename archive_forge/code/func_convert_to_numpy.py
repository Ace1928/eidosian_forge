from collections import OrderedDict
from gymnasium.spaces import Discrete, MultiDiscrete
import numpy as np
import tree  # pip install dm_tree
from types import MappingProxyType
from typing import List, Optional
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.typing import SpaceStruct, TensorType, TensorStructType, Union
@PublicAPI
def convert_to_numpy(x: TensorStructType, reduce_type: bool=True, reduce_floats=DEPRECATED_VALUE):
    """Converts values in `stats` to non-Tensor numpy or python types.

    Args:
        x: Any (possibly nested) struct, the values in which will be
            converted and returned as a new struct with all torch/tf tensors
            being converted to numpy types.
        reduce_type: Whether to automatically reduce all float64 and int64 data
            into float32 and int32 data, respectively.

    Returns:
        A new struct with the same structure as `x`, but with all
        values converted to numpy arrays (on CPU).
    """
    if reduce_floats != DEPRECATED_VALUE:
        deprecation_warning(old='reduce_floats', new='reduce_types', error=True)
        reduce_type = reduce_floats

    def mapping(item):
        if torch and isinstance(item, torch.Tensor):
            ret = item.cpu().item() if len(item.size()) == 0 else item.detach().cpu().numpy()
        elif tf and isinstance(item, (tf.Tensor, tf.Variable)) and hasattr(item, 'numpy'):
            assert tf.executing_eagerly()
            ret = item.numpy()
        else:
            ret = item
        if reduce_type and isinstance(ret, np.ndarray):
            if np.issubdtype(ret.dtype, np.floating):
                ret = ret.astype(np.float32)
            elif np.issubdtype(ret.dtype, int):
                ret = ret.astype(np.int32)
            return ret
        return ret
    return tree.map_structure(mapping, x)
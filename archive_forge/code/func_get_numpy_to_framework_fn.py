import io
import pathlib
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils import BaseImageProcessor, get_size_dict
from ...image_transforms import (
from ...image_utils import (
from ...utils import (
def get_numpy_to_framework_fn(arr) -> Callable:
    """
    Returns a function that converts a numpy array to the framework of the input array.

    Args:
        arr (`np.ndarray`): The array to convert.
    """
    if isinstance(arr, np.ndarray):
        return np.array
    if is_tf_available() and is_tf_tensor(arr):
        import tensorflow as tf
        return tf.convert_to_tensor
    if is_torch_available() and is_torch_tensor(arr):
        import torch
        return torch.tensor
    if is_flax_available() and is_jax_tensor(arr):
        import jax.numpy as jnp
        return jnp.array
    raise ValueError(f'Cannot convert arrays of type {type(arr)}')
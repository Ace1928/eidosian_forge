from typing import List, Optional, Union
import numpy as np
import tensorflow as tf
from .feature_extraction_utils import BatchFeature
from .tokenization_utils_base import BatchEncoding
from .utils import logging
def check_embeddings_within_bounds(tensor: tf.Tensor, embed_dim: int, tensor_name: str='input_ids') -> None:
    """
    `tf.gather`, on which TF embedding layers are based, won't check positive out of bound indices on GPU, returning
    zeros instead. This function adds a check against that dangerous silent behavior.

    Args:
        tensor (`tf.Tensor`): The tensor of indices to check.
        embed_dim (`int`): The embedding dimension.
        tensor_name (`str`, *optional*): The name of the tensor to use in the error message.
    """
    tf.debugging.assert_less(tensor, tf.cast(embed_dim, dtype=tensor.dtype), message=f"The maximum value of {tensor_name} ({tf.math.reduce_max(tensor)}) must be smaller than the embedding layer's input dimension ({embed_dim}). The likely cause is some problem at tokenization time.")
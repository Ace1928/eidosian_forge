import os
from typing import Union, Optional, Tuple, Any, List, Sized, TypeVar
import itertools
from collections import namedtuple
import parlai.utils.logging as logging
import torch.optim
@staticmethod
def chunk_to(chunk: Chunk, device: str) -> Chunk:
    """
        Move the chunk to the device.

        Handles chunks which are groups of tensors.
        """
    if isinstance(chunk, torch.Tensor):
        return chunk.to(device)
    elif isinstance(chunk, tuple):
        return tuple((PipelineHelper.chunk_to(c, device) for c in chunk))
    elif isinstance(chunk, dict):
        return {k: PipelineHelper.chunk_to(v, device) for k, v in chunk.items()}
    else:
        raise TypeError('chunk_to only compatible with tensors, tuples or dicts.')
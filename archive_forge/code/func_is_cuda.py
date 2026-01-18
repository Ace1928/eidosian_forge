from contextlib import contextmanager
from typing import Generator, List, Union, cast
import torch
def is_cuda(stream: AbstractStream) -> bool:
    """Returns ``True`` if the given stream is a valid CUDA stream."""
    return stream is not CPUStream
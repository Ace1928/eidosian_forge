from contextlib import contextmanager
from typing import Generator, List, Union, cast
import torch
def as_cuda(stream: AbstractStream) -> torch.cuda.Stream:
    """Casts the given stream as :class:`torch.cuda.Stream`."""
    return cast(torch.cuda.Stream, stream)
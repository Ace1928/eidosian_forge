from contextlib import contextmanager
from typing import Generator, List, Union, cast
import torch
def new_stream(device: torch.device) -> AbstractStream:
    """Creates a new stream for either CPU or CUDA device."""
    if device.type != 'cuda':
        return CPUStream
    return torch.cuda.Stream(device)
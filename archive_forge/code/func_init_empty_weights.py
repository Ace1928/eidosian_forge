import logging
import os
from contextlib import contextmanager
from functools import wraps
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
from .hooks import (
from .utils import (
from .utils.other import recursive_getattr
@contextmanager
def init_empty_weights(include_buffers: bool=None):
    """
    A context manager under which models are initialized with all parameters on the meta device, therefore creating an
    empty model. Useful when just initializing the model would blow the available RAM.

    Args:
        include_buffers (`bool`, *optional*):
            Whether or not to also put all buffers on the meta device while initializing.

    Example:

    ```python
    import torch.nn as nn
    from accelerate import init_empty_weights

    # Initialize a model with 100 billions parameters in no time and without using any RAM.
    with init_empty_weights():
        tst = nn.Sequential(*[nn.Linear(10000, 10000) for _ in range(1000)])
    ```

    <Tip warning={true}>

    Any model created under this context manager has no weights. As such you can't do something like
    `model.to(some_device)` with it. To load weights inside your empty model, see [`load_checkpoint_and_dispatch`].
    Make sure to overwrite the default device_map param for [`load_checkpoint_and_dispatch`], otherwise dispatch is not
    called.

    </Tip>
    """
    if include_buffers is None:
        include_buffers = parse_flag_from_env('ACCELERATE_INIT_INCLUDE_BUFFERS', False)
    with init_on_device(torch.device('meta'), include_buffers=include_buffers) as f:
        yield f
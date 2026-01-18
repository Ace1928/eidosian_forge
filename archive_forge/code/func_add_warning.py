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
def add_warning(fn, model):

    @wraps(fn)
    def wrapper(*args, **kwargs):
        warning_msg = "You shouldn't move a model that is dispatched using accelerate hooks."
        if str(fn.__name__) == 'to':
            to_device = torch._C._nn._parse_to(*args, **kwargs)[0]
            if to_device is not None:
                logger.warning(warning_msg)
        else:
            logger.warning(warning_msg)
        for param in model.parameters():
            if param.device == torch.device('meta'):
                raise RuntimeError("You can't move a model that has some modules offloaded to cpu or disk.")
        return fn(*args, **kwargs)
    return wrapper
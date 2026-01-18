import contextlib
import torch
from torch._C._functorch import (
from typing import Union, Tuple
def exposed_in(module):

    def wrapper(fn):
        fn.__module__ = module
        return fn
    return wrapper
from collections import namedtuple
from typing import Optional, Any, Union, Type
import torch
import torch.nn as nn
from torch.ao.quantization.fake_quantize import (
from .observer import (
import warnings
import copy
def configure_constructor_to_put_obs_on_module_device(original_constructor):
    try:
        check = original_constructor.with_args(factory_kwargs=None)
        check()
        return original_constructor.with_callable_args(factory_kwargs=get_factory_kwargs_based_on_module_device)
    except AttributeError:
        return original_constructor
    except TypeError:
        return original_constructor
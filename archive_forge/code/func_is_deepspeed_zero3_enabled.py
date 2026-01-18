import copy
import importlib.metadata as importlib_metadata
import importlib.util
import weakref
from functools import partialmethod
from ..dependency_versions_check import dep_version_check
from ..utils import is_accelerate_available, is_torch_available, logging
def is_deepspeed_zero3_enabled():
    if _hf_deepspeed_config_weak_ref is not None and _hf_deepspeed_config_weak_ref() is not None:
        return _hf_deepspeed_config_weak_ref().is_zero3()
    else:
        return False
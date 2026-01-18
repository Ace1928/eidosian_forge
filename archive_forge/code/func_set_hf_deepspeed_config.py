import copy
import importlib.metadata as importlib_metadata
import importlib.util
import weakref
from functools import partialmethod
from ..dependency_versions_check import dep_version_check
from ..utils import is_accelerate_available, is_torch_available, logging
def set_hf_deepspeed_config(hf_deepspeed_config_obj):
    global _hf_deepspeed_config_weak_ref
    _hf_deepspeed_config_weak_ref = weakref.ref(hf_deepspeed_config_obj)
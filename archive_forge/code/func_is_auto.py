import copy
import importlib.metadata as importlib_metadata
import importlib.util
import weakref
from functools import partialmethod
from ..dependency_versions_check import dep_version_check
from ..utils import is_accelerate_available, is_torch_available, logging
def is_auto(self, ds_key_long):
    val = self.get_value(ds_key_long)
    if val is None:
        return False
    else:
        return val == 'auto'
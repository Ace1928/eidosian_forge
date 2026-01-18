from ._abc import Loader
from ._bootstrap import module_from_spec
from ._bootstrap import _resolve_name
from ._bootstrap import spec_from_loader
from ._bootstrap import _find_spec
from ._bootstrap_external import MAGIC_NUMBER
from ._bootstrap_external import _RAW_MAGIC_NUMBER
from ._bootstrap_external import cache_from_source
from ._bootstrap_external import decode_source
from ._bootstrap_external import source_from_cache
from ._bootstrap_external import spec_from_file_location
from contextlib import contextmanager
import _imp
import functools
import sys
import types
import warnings
@functools.wraps(fxn)
def module_for_loader_wrapper(self, fullname, *args, **kwargs):
    with _module_to_load(fullname) as module:
        module.__loader__ = self
        try:
            is_package = self.is_package(fullname)
        except (ImportError, AttributeError):
            pass
        else:
            if is_package:
                module.__package__ = fullname
            else:
                module.__package__ = fullname.rpartition('.')[0]
        return fxn(self, module, *args, **kwargs)
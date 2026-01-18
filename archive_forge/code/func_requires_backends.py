import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import warnings
from collections import OrderedDict
from functools import lru_cache
from itertools import chain
from types import ModuleType
from typing import Any, Tuple, Union
from packaging import version
from . import logging
def requires_backends(obj, backends):
    if not isinstance(backends, (list, tuple)):
        backends = [backends]
    name = obj.__name__ if hasattr(obj, '__name__') else obj.__class__.__name__
    if 'torch' in backends and 'tf' not in backends and (not is_torch_available()) and is_tf_available():
        raise ImportError(PYTORCH_IMPORT_ERROR_WITH_TF.format(name))
    if 'tf' in backends and 'torch' not in backends and is_torch_available() and (not is_tf_available()):
        raise ImportError(TF_IMPORT_ERROR_WITH_PYTORCH.format(name))
    checks = (BACKENDS_MAPPING[backend] for backend in backends)
    failed = [msg.format(name) for available, msg in checks if not available()]
    if failed:
        raise ImportError(''.join(failed))
import os
import sys
import hashlib
import filecmp
from collections import namedtuple
from shutil import which
from jedi.cache import memoize_method, time_cache
from jedi.inference.compiled.subprocess import CompiledSubprocess, \
import parso
def get_cached_default_environment():
    var = os.environ.get('VIRTUAL_ENV') or os.environ.get(_CONDA_VAR)
    environment = _get_cached_default_environment()
    if var and os.path.realpath(var) != os.path.realpath(environment.path):
        _get_cached_default_environment.clear_cache()
        return _get_cached_default_environment()
    return environment
import functools
import inspect
import sys
import unittest
from tensorflow.python.autograph.core import config
from tensorflow.python.autograph.pyct import cache
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.autograph.utils import ag_logging as logging
from tensorflow.python.eager.polymorphic_function import tf_method_target
from tensorflow.python.util import tf_inspect
def is_in_allowlist_cache(entity, options):
    try:
        return _ALLOWLIST_CACHE.has(entity, options)
    except TypeError:
        return False
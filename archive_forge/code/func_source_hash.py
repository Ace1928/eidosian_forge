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
def source_hash(source_bytes):
    """Return the hash of *source_bytes* as used in hash-based pyc files."""
    return _imp.source_hash(_RAW_MAGIC_NUMBER, source_bytes)
import sys
import os
from six import iteritems
from enum import IntEnum
from contextlib import contextmanager
import json
def get_hnsw_bin_module():
    if '_hnsw' in sys.modules:
        return sys.modules['_hnsw']
    so_paths = get_so_paths('./')
    for so_path in so_paths:
        try:
            loaded_hnsw = load_dynamic('_hnsw', so_path)
            sys.modules['hnsw._hnsw'] = loaded_hnsw
            return loaded_hnsw
        except ImportError:
            pass
    from . import _hnsw
    return _hnsw
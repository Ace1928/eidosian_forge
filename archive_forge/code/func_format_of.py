import hashlib
import os
import tempfile
from ..common import _build
from ..common.backend import get_cuda_version_key
from ..common.build import is_hip
from ..runtime.cache import get_cache_manager
from .utils import generate_cu_signature
def format_of(ty):
    return {'PyObject*': 'O', 'float': 'f', 'double': 'd', 'long': 'l', 'uint32_t': 'I', 'int32_t': 'i', 'uint64_t': 'K', 'int64_t': 'L'}[ty]
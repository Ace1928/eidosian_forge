import hashlib
import os
import tempfile
from ..common import _build
from ..common.backend import get_cuda_version_key
from ..common.build import is_hip
from ..runtime.cache import get_cache_manager
from .utils import generate_cu_signature
def make_so_cache_key(version_hash, signature, constants, ids, **kwargs):
    signature = {k: 'ptr' if v[0] == '*' else v for k, v in signature.items()}
    key = f'{version_hash}-{''.join(signature.values())}-{constants}-{ids}'
    for kw in kwargs:
        key = f'{key}-{kwargs.get(kw)}'
    key = hashlib.md5(key.encode('utf-8')).hexdigest()
    return key
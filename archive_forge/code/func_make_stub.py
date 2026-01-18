import hashlib
import os
import tempfile
from ..common import _build
from ..common.backend import get_cuda_version_key
from ..common.build import is_hip
from ..runtime.cache import get_cache_manager
from .utils import generate_cu_signature
def make_stub(name, signature, constants, ids, **kwargs):
    so_cache_key = make_so_cache_key(get_cuda_version_key(), signature, constants, ids, **kwargs)
    so_cache_manager = get_cache_manager(so_cache_key)
    so_name = f'{name}.so'
    cache_path = so_cache_manager.get_file(so_name)
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src = generate_launcher(constants, signature, ids)
            src_path = os.path.join(tmpdir, 'main.c')
            with open(src_path, 'w') as f:
                f.write(src)
            so = _build(name, src_path, tmpdir)
            with open(so, 'rb') as f:
                return so_cache_manager.put(f.read(), so_name, binary=True)
    else:
        return cache_path
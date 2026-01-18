import difflib
import os
import io
import shutil
import struct
import sys
import torch
import tarfile
import tempfile
import warnings
from contextlib import closing, contextmanager
from enum import Enum
from ._utils import _import_dotted_name
from torch._sources import get_source_lines_and_file
from torch.types import Storage
from torch.storage import _get_dtype_from_pickle_storage_type
from typing import Any, BinaryIO, Callable, cast, Dict, Optional, Type, Tuple, Union, IO
from typing_extensions import TypeAlias  # Python 3.10+
import copyreg
import pickle
import pathlib
import torch._weights_only_unpickler as _weights_only_unpickler
def legacy_load(f):
    deserialized_objects: Dict[int, Any] = {}

    def persistent_load(saved_id):
        if isinstance(saved_id, tuple):
            if all(saved_id[1:]):
                _check_container_source(*saved_id)
            return saved_id[0]
        return deserialized_objects[int(saved_id)]
    with closing(tarfile.open(fileobj=f, mode='r:', format=tarfile.PAX_FORMAT)) as tar, mkdtemp() as tmpdir:
        tar.extract('storages', path=tmpdir)
        with open(os.path.join(tmpdir, 'storages'), 'rb', 0) as f:
            num_storages = pickle_module.load(f, **pickle_load_args)
            for i in range(num_storages):
                args = pickle_module.load(f, **pickle_load_args)
                key, location, storage_type = args
                dtype = storage_type._dtype
                obj = cast(Storage, torch.UntypedStorage)._new_with_file(f, torch._utils._element_size(dtype))
                obj = restore_location(obj, location)
                deserialized_objects[key] = torch.storage.TypedStorage(wrap_storage=obj, dtype=dtype, _internal=True)
            storage_views = pickle_module.load(f, **pickle_load_args)
            for target_cdata, root_cdata, offset, numel in storage_views:
                root = deserialized_objects[root_cdata]
                element_size = torch._utils._element_size(root.dtype)
                offset_bytes = offset * element_size
                deserialized_objects[target_cdata] = torch.storage.TypedStorage(wrap_storage=root._untyped_storage[offset_bytes:offset_bytes + numel * element_size], dtype=root.dtype, _internal=True)
        tar.extract('tensors', path=tmpdir)
        with open(os.path.join(tmpdir, 'tensors'), 'rb', 0) as f:
            num_tensors = pickle_module.load(f, **pickle_load_args)
            for _ in range(num_tensors):
                args = pickle_module.load(f, **pickle_load_args)
                key, storage_id, original_tensor_type = args
                storage = deserialized_objects[storage_id]
                ndim, = struct.unpack('<i', f.read(4))
                f.read(4)
                numel = struct.unpack(f'<{ndim}q', f.read(8 * ndim))
                stride = struct.unpack(f'<{ndim}q', f.read(8 * ndim))
                storage_offset, = struct.unpack('<q', f.read(8))
                tensor = torch.tensor([], dtype=storage.dtype).set_(storage._untyped_storage, storage_offset, numel, stride)
                deserialized_objects[key] = tensor
        pickle_file = tar.extractfile('pickle')
        unpickler = UnpicklerWrapper(pickle_file, **pickle_load_args)
        unpickler.persistent_load = persistent_load
        result = unpickler.load()
        return result
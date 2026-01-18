from collections.abc import MutableMapping
from pathlib import Path
from typing import Literal, TypeVar, Union
from uuid import uuid4
from numpy.typing import ArrayLike
from ._lazy_modules import fsspec, h5py
def open_hdf5_s3(s3_url: str, *, block_size: int=8388608) -> HDF5Group:
    """Uses ``fsspec`` module to open the HDF5 file at ``s3_url``.

    This requires both ``fsspec`` and ``aiohttp`` to be installed.

    Args:
        s3_url: URL of dataset file in S3
        block_size: Number of bytes to fetch per read operation. Larger values
            may improve performance for large datasets
    """
    memory_cache_args = {'cache_type': 'mmap', 'block_size': block_size}
    fs = fsspec.open(s3_url, **memory_cache_args)
    return h5py.File(fs.open())
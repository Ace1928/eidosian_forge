from collections.abc import MutableMapping
from pathlib import Path
from typing import Literal, TypeVar, Union
from uuid import uuid4
from numpy.typing import ArrayLike
from ._lazy_modules import fsspec, h5py
def copy_all(source: HDF5Group, dest: HDF5Group, *keys: str, on_conflict: Literal['raise', 'overwrite', 'ignore']='ignore', without_attrs: bool=False) -> None:
    """Copies all the elements of ``source`` named ``keys`` into ``dest``. If no keys
    are provided, all elements of ``source`` will be copied."""
    if not keys:
        keys = source
    for key in keys:
        copy(source[key], dest, key=key, on_conflict=on_conflict)
    if not without_attrs:
        dest.attrs.update(source.attrs)
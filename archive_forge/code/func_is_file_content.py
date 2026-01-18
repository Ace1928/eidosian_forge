from __future__ import annotations
import io
import os
import pathlib
from typing import overload
from typing_extensions import TypeGuard
import anyio
from ._types import (
from ._utils import is_tuple_t, is_mapping_t, is_sequence_t
def is_file_content(obj: object) -> TypeGuard[FileContent]:
    return isinstance(obj, bytes) or isinstance(obj, tuple) or isinstance(obj, io.IOBase) or isinstance(obj, os.PathLike)
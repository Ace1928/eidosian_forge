from __future__ import annotations
import io
import os
import pathlib
from typing import overload
from typing_extensions import TypeGuard
import anyio
from ._types import (
from ._utils import is_tuple_t, is_mapping_t, is_sequence_t
def assert_is_file_content(obj: object, *, key: str | None=None) -> None:
    if not is_file_content(obj):
        prefix = f'Expected entry at `{key}`' if key is not None else f'Expected file input `{obj!r}`'
        raise RuntimeError(f'{prefix} to be bytes, an io.IOBase instance, PathLike or a tuple but received {type(obj)} instead. See https://github.com/openai/openai-python/tree/main#file-uploads') from None
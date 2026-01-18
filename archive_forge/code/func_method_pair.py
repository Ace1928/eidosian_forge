from __future__ import annotations
import os
import pathlib
from typing import TYPE_CHECKING, Type, Union
import pytest
import trio
from trio._file_io import AsyncIOWrapper
def method_pair(path: str, method_name: str) -> tuple[Callable[[], object], Callable[[], Awaitable[object]]]:
    sync_path = pathlib.Path(path)
    async_path = trio.Path(path)
    return (getattr(sync_path, method_name), getattr(async_path, method_name))
import io
import os
import pathlib
import sys
from typing import IO, Any, BinaryIO, List, Tuple
import trio
from trio._file_io import AsyncIOWrapper
from typing_extensions import assert_type
def operator_checks(text: str, tpath: trio.Path, ppath: pathlib.Path) -> None:
    """Verify operators produce the right results."""
    assert_type(tpath / ppath, trio.Path)
    assert_type(tpath / tpath, trio.Path)
    assert_type(tpath / text, trio.Path)
    assert_type(text / tpath, trio.Path)
    assert_type(tpath > tpath, bool)
    assert_type(tpath >= tpath, bool)
    assert_type(tpath < tpath, bool)
    assert_type(tpath <= tpath, bool)
    assert_type(tpath > ppath, bool)
    assert_type(tpath >= ppath, bool)
    assert_type(tpath < ppath, bool)
    assert_type(tpath <= ppath, bool)
    assert_type(ppath > tpath, bool)
    assert_type(ppath >= tpath, bool)
    assert_type(ppath < tpath, bool)
    assert_type(ppath <= tpath, bool)
from __future__ import annotations
import enum
import re
from typing import TYPE_CHECKING, NewType, NoReturn, Protocol, cast
import cffi
def raise_winerror(winerror: int | None=None, *, filename: str | None=None, filename2: str | None=None) -> NoReturn:
    if winerror is None:
        err = ffi.getwinerror()
        if err is None:
            raise RuntimeError('No error set?')
        winerror, msg = err
    else:
        err = ffi.getwinerror(winerror)
        if err is None:
            raise RuntimeError('No error set?')
        _, msg = err
    raise OSError(0, msg, filename, winerror, filename2)
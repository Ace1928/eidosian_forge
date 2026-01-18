import os
import pathlib
import platform
import stat
import sys
from logging import getLogger
from typing import Union
def is_reparse_point(path: Union[str, pathlib.Path]) -> bool:
    GetFileAttributesW.argtypes = [LPCWSTR]
    GetFileAttributesW.restype = DWORD
    return _check_bit(GetFileAttributesW(str(path)), stat.FILE_ATTRIBUTE_REPARSE_POINT)
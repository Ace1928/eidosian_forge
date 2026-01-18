import os
import sys
import ctypes
import pathlib
import platform
from typing import Optional, List, Tuple, Callable
def load_rwkv_shared_library() -> RWKVSharedLibrary:
    """
    Attempts to find rwkv.cpp shared library and load it.
    To specify exact path to the library, create an instance of RWKVSharedLibrary explicitly.
    """
    file_name: str
    if 'win32' in sys.platform or 'cygwin' in sys.platform:
        file_name = 'rwkv.dll'
    elif 'darwin' in sys.platform:
        file_name = 'librwkv.dylib'
    else:
        file_name = 'librwkv.so'
    child_paths: List[Callable[[pathlib.Path], pathlib.Path]] = [lambda p: p / 'backend-python' / 'rwkv_pip' / 'cpp' / file_name, lambda p: p / 'bin' / 'Release' / file_name, lambda p: p / 'bin' / file_name, lambda p: p / 'build' / 'bin' / 'Release' / file_name, lambda p: p / 'build' / 'bin' / file_name, lambda p: p / 'build' / file_name, lambda p: p / file_name]
    working_dir: pathlib.Path = pathlib.Path(os.path.abspath(os.getcwd()))
    parent_paths: List[pathlib.Path] = [working_dir.parent.parent, working_dir.parent, working_dir, pathlib.Path(os.path.abspath(__file__)).parent.parent.parent]
    for parent_path in parent_paths:
        for child_path in child_paths:
            full_path: pathlib.Path = child_path(parent_path)
            if os.path.isfile(full_path):
                return RWKVSharedLibrary(str(full_path))
    raise ValueError(f'Failed to find {file_name} automatically; you need to find the library and create RWKVSharedLibrary specifying the path to it')
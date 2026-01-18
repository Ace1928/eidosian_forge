import os
import sys
import shlex
import importlib
import subprocess
import pkg_resources
import pathlib
from typing import List, Type, Any, Union, Dict
from types import ModuleType
@classmethod
def import_cmd(cls, binary: str, resolve_missing: bool=True, require: bool=False, flags: List[str]=None):
    """ Lazily builds a lazy.Cmd based on binary
        
            if available, returns the lazy.Cmd(binary)
            if missing and resolve_missing = True, will lazily install in host system
        else:
            if require: raise ImportError
            returns None
        """
    if not cls.is_exec_available(binary):
        if require and (not resolve_missing):
            raise ImportError(f'Required Executable {binary} is not available.')
        if not resolve_missing:
            return None
        cls.install_binary(binary, flags=flags)
    from lazy.cmd import Cmd
    return Cmd(binary=binary)
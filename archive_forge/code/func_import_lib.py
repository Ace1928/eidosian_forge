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
def import_lib(cls, library: str, pip_name: str=None, resolve_missing: bool=True, require: bool=False, upgrade: bool=False) -> ModuleType:
    """ Lazily resolves libs.

            if pip_name is provided, will install using pip_name, otherwise will use libraryname

            ie ->   LazyLib.import_lib('fuse', 'fusepy') # if fusepy is not expected to be available, and fusepy is the pip_name
                    LazyLib.import_lib('fuse') # if fusepy is expected to be available
            
            returns `fuse` as if you ran `import fuse`
        
            if available, returns the sys.modules[library]
            if missing and resolve_missing = True, will lazily install
        else:
            if require: raise ImportError
            returns None
        """
    clean_lib = cls.get_requirement(library, True)
    if not cls.is_available(clean_lib):
        if require and (not resolve_missing):
            raise ImportError(f'Required Lib {library} is not available.')
        if not resolve_missing:
            return None
        cls.install_library(pip_name or library, upgrade=upgrade)
    return cls._ensure_lib_imported(library)
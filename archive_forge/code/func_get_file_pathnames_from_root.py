import fnmatch
import functools
import inspect
import os
import warnings
from io import IOBase
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from torch.utils.data._utils.serialization import DILL_AVAILABLE
def get_file_pathnames_from_root(root: str, masks: Union[str, List[str]], recursive: bool=False, abspath: bool=False, non_deterministic: bool=False) -> Iterable[str]:

    def onerror(err: OSError):
        warnings.warn(err.filename + ' : ' + err.strerror)
        raise err
    if os.path.isfile(root):
        path = root
        if abspath:
            path = os.path.abspath(path)
        fname = os.path.basename(path)
        if match_masks(fname, masks):
            yield path
    else:
        for path, dirs, files in os.walk(root, onerror=onerror):
            if abspath:
                path = os.path.abspath(path)
            if not non_deterministic:
                files.sort()
            for f in files:
                if match_masks(f, masks):
                    yield os.path.join(path, f)
            if not recursive:
                break
            if not non_deterministic:
                dirs.sort()
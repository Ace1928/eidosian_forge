import glob
import importlib
import inspect
import logging
import os
import re
import sys
from typing import Iterable, List, Optional, Tuple, Union
def find_source() -> Tuple[str, int, int]:
    obj = sys.modules[info['module']]
    for part in info['fullname'].split('.'):
        obj = getattr(obj, part)
    fname = str(inspect.getsourcefile(obj))
    if any((s in fname for s in ('readthedocs', 'rtfd', 'checkouts'))):
        path_top = os.path.abspath(os.path.join('..', '..', '..'))
        fname = str(os.path.relpath(fname, start=path_top))
    else:
        fname = f'master/{os.path.relpath(fname, start=os.path.abspath('..'))}'
    source, line_start = inspect.getsourcelines(obj)
    return (fname, line_start, line_start + len(source) - 1)
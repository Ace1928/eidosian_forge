import glob
import inspect
import os
import re
import sys
from importlib import import_module
from importlib.machinery import all_suffixes
from time import time
from zipimport import zipimporter
from .completer import expand_user, compress_user
from .error import TryNext
from ..utils._process_common import arg_split
from IPython import get_ipython
from typing import List
import_re = re.compile(r'(?P<name>[^\W\d]\w*?)'
def get_root_modules():
    """
    Returns a list containing the names of all the modules available in the
    folders of the pythonpath.

    ip.db['rootmodules_cache'] maps sys.path entries to list of modules.
    """
    ip = get_ipython()
    if ip is None:
        return list(sys.builtin_module_names)
    if getattr(ip.db, '_mock', False):
        rootmodules_cache = {}
    else:
        rootmodules_cache = ip.db.get('rootmodules_cache', {})
    rootmodules = list(sys.builtin_module_names)
    start_time = time()
    store = False
    for path in sys.path:
        try:
            modules = rootmodules_cache[path]
        except KeyError:
            modules = module_list(path)
            try:
                modules.remove('__init__')
            except ValueError:
                pass
            if path not in ('', '.'):
                rootmodules_cache[path] = modules
            if time() - start_time > TIMEOUT_STORAGE and (not store):
                store = True
                print('\nCaching the list of root modules, please wait!')
                print("(This will only be done once - type '%rehashx' to reset cache!)\n")
                sys.stdout.flush()
            if time() - start_time > TIMEOUT_GIVEUP:
                print('This is taking too long, we give up.\n')
                return []
        rootmodules.extend(modules)
    if store:
        ip.db['rootmodules_cache'] = rootmodules_cache
    rootmodules = list(set(rootmodules))
    return rootmodules
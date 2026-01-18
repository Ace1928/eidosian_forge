from __future__ import print_function
import gc
import inspect
import os
import sys
import traceback
import types
from .debug import printExc
def reloadAll(prefix=None, debug=False):
    """Automatically reload all modules whose __file__ begins with *prefix*.

    Skips reload if the file has not been updated (if .pyc is newer than .py)
    If *prefix* is None, then all loaded modules are checked.

    Returns a dictionary {moduleName: (reloaded, reason)} describing actions taken
    for each module.
    """
    failed = []
    changed = []
    ret = {}
    for modName, mod in list(sys.modules.items()):
        if not inspect.ismodule(mod):
            ret[modName] = (False, 'not a module')
            continue
        if modName == '__main__':
            ret[modName] = (False, 'ignored __main__')
            continue
        if getattr(mod, '__file__', None) is None:
            ret[modName] = (False, 'module has no __file__')
            continue
        if os.path.splitext(mod.__file__)[1] not in ['.py', '.pyc']:
            ret[modName] = (False, '%s not a .py/pyc file' % str(mod.__file__))
            continue
        if prefix is not None and mod.__file__[:len(prefix)] != prefix:
            ret[modName] = (False, 'file %s not in prefix %s' % (mod.__file__, prefix))
            continue
        py = os.path.splitext(mod.__file__)[0] + '.py'
        if py in changed:
            continue
        if not os.path.isfile(py):
            ret[modName] = (False, '.py does not exist: %s' % py)
            continue
        pyc = getattr(mod, '__cached__', py + 'c')
        if not os.path.isfile(pyc):
            ret[modName] = (False, 'code has no pyc file to compare')
            continue
        if os.stat(pyc).st_mtime > os.stat(py).st_mtime:
            ret[modName] = (False, 'code has not changed since compile')
            continue
        changed.append(py)
        try:
            reload(mod, debug=debug)
            ret[modName] = (True, None)
        except Exception as exc:
            printExc('Error while reloading module %s, skipping\n' % mod)
            failed.append(mod.__name__)
            ret[modName] = (False, 'reload failed: %s' % traceback.format_exception_only(type(exc), exc))
    if len(failed) > 0:
        raise Exception('Some modules failed to reload: %s' % ', '.join(failed))
    return ret
from collections import namedtuple
from hashlib import sha256
import os
import shutil
import sys
import fnmatch
from sympy.testing.pytest import XFAIL
def import_module_from_file(filename, only_if_newer_than=None):
    """ Imports Python extension (from shared object file)

    Provide a list of paths in `only_if_newer_than` to check
    timestamps of dependencies. import_ raises an ImportError
    if any is newer.

    Word of warning: The OS may cache shared objects which makes
    reimporting same path of an shared object file very problematic.

    It will not detect the new time stamp, nor new checksum, but will
    instead silently use old module. Use unique names for this reason.

    Parameters
    ==========

    filename : str
        Path to shared object.
    only_if_newer_than : iterable of strings
        Paths to dependencies of the shared object.

    Raises
    ======

    ``ImportError`` if any of the files specified in ``only_if_newer_than`` are newer
    than the file given by filename.
    """
    path, name = os.path.split(filename)
    name, ext = os.path.splitext(name)
    name = name.split('.')[0]
    if sys.version_info[0] == 2:
        from imp import find_module, load_module
        fobj, filename, data = find_module(name, [path])
        if only_if_newer_than:
            for dep in only_if_newer_than:
                if os.path.getmtime(filename) < os.path.getmtime(dep):
                    raise ImportError('{} is newer than {}'.format(dep, filename))
        mod = load_module(name, fobj, filename, data)
    else:
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, filename)
        if spec is None:
            raise ImportError("Failed to import: '%s'" % filename)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod
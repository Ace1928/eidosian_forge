import os
import sys
import errno
import shutil
import random
import glob
import warnings
from IPython.utils.process import system
def get_py_filename(name):
    """Return a valid python filename in the current directory.

    If the given name is not a file, it adds '.py' and searches again.
    Raises IOError with an informative message if the file isn't found.
    """
    name = os.path.expanduser(name)
    if os.path.isfile(name):
        return name
    if not name.endswith('.py'):
        py_name = name + '.py'
        if os.path.isfile(py_name):
            return py_name
    raise IOError('File `%r` not found.' % name)
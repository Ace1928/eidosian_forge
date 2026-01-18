from . import VENDORED_ROOT
from ._util import cwd, iter_all_files
def prune_dir(dirname, basename):
    if basename == '__pycache__':
        return True
    elif dirname != 'pydevd':
        return False
    elif basename.startswith('pydev'):
        return False
    elif basename.startswith('_pydev'):
        return False
    return True
import os
import stat
from ..osutils import pathjoin
from ..trace import warning
def capture_tree_contents(top):
    """Make a Python datastructure description of a tree.

    If top is an absolute path the descriptions will be absolute."""
    for dirpath, dirnames, filenames in os.walk(top):
        yield (dirpath + '/',)
        filenames.sort()
        for fn in filenames:
            fullpath = pathjoin(dirpath, fn)
            if fullpath[-1] in '@/':
                raise AssertionError(fullpath)
            info = os.lstat(fullpath)
            if stat.S_ISLNK(info.st_mode):
                yield (fullpath + '@', os.readlink(fullpath))
            elif stat.S_ISREG(info.st_mode):
                with open(fullpath, 'rb') as f:
                    file_bytes = f.read()
                yield (fullpath, file_bytes)
            else:
                warning("can't capture file %s with mode %#o", fullpath, info.st_mode)
                pass
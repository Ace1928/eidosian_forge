import sys
import os
import builtins
import _sitebuiltins
import io
import stat
def makepath(*paths):
    dir = os.path.join(*paths)
    try:
        dir = os.path.abspath(dir)
    except OSError:
        pass
    return (dir, os.path.normcase(dir))
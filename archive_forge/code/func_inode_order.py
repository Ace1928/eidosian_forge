import os
import stat
import time
from .. import atomicfile, errors
from .. import filters as _mod_filters
from .. import osutils, trace
def inode_order(path_and_cache):
    return path_and_cache[1][1][3]
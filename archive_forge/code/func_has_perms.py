import importlib
import os
import stat
import subprocess
import sys
import tempfile
import warnings
from .. import osutils, symbol_versioning
def has_perms():
    write_perms = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
    f = tempfile.mkstemp(prefix='bzr_perms_chk_')
    fd, name = f
    os.close(fd)
    osutils.chmod_if_possible(name, write_perms)
    read_perms = os.stat(name).st_mode & 511
    os.unlink(name)
    return write_perms == read_perms
import os
import sys
import stat
import fnmatch
import collections
import errno
def copymode(src, dst, *, follow_symlinks=True):
    """Copy mode bits from src to dst.

    If follow_symlinks is not set, symlinks aren't followed if and only
    if both `src` and `dst` are symlinks.  If `lchmod` isn't available
    (e.g. Linux) this method does nothing.

    """
    sys.audit('shutil.copymode', src, dst)
    if not follow_symlinks and _islink(src) and os.path.islink(dst):
        if os.name == 'nt':
            stat_func, chmod_func = (os.lstat, os.chmod)
        elif hasattr(os, 'lchmod'):
            stat_func, chmod_func = (os.lstat, os.lchmod)
        else:
            return
    else:
        if os.name == 'nt' and os.path.islink(dst):
            dst = os.path.realpath(dst, strict=True)
        stat_func, chmod_func = (_stat, os.chmod)
    st = stat_func(src)
    chmod_func(dst, stat.S_IMODE(st.st_mode))
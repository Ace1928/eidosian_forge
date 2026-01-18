from __future__ import print_function, unicode_literals
import typing
import warnings
from .errors import ResourceNotFound
from .opener import manage_fs
from .path import abspath, combine, frombase, normpath
from .tools import is_thread_safe
from .walk import Walker
def copy_file_if_newer(src_fs, src_path, dst_fs, dst_path, preserve_time=False):
    """Copy a file from one filesystem to another, checking times.

    .. deprecated:: 2.5.0
       Use `~fs.copy.copy_file_if` with ``condition="newer"`` instead.

    """
    warnings.warn('copy_file_if_newer is deprecated. Use copy_file_if instead.', DeprecationWarning)
    return copy_file_if(src_fs, src_path, dst_fs, dst_path, 'newer', preserve_time=preserve_time)
from __future__ import print_function, unicode_literals
import typing
import warnings
from .errors import ResourceNotFound
from .opener import manage_fs
from .path import abspath, combine, frombase, normpath
from .tools import is_thread_safe
from .walk import Walker
def copy_modified_time(src_fs, src_path, dst_fs, dst_path):
    """Copy modified time metadata from one file to another.

    Arguments:
        src_fs (FS or str): Source filesystem (instance or URL).
        src_path (str): Path to a directory on the source filesystem.
        dst_fs (FS or str): Destination filesystem (instance or URL).
        dst_path (str): Path to a directory on the destination filesystem.

    """
    namespaces = ('details',)
    with manage_fs(src_fs, writeable=False) as _src_fs:
        with manage_fs(dst_fs, create=True) as _dst_fs:
            src_meta = _src_fs.getinfo(src_path, namespaces)
            src_details = src_meta.raw.get('details', {})
            dst_details = {}
            for value in ('metadata_changed', 'modified'):
                if value in src_details:
                    dst_details[value] = src_details[value]
            _dst_fs.setinfo(dst_path, {'details': dst_details})
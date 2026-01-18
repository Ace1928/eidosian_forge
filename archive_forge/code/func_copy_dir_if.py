from __future__ import print_function, unicode_literals
import typing
import warnings
from .errors import ResourceNotFound
from .opener import manage_fs
from .path import abspath, combine, frombase, normpath
from .tools import is_thread_safe
from .walk import Walker
def copy_dir_if(src_fs, src_path, dst_fs, dst_path, condition, walker=None, on_copy=None, workers=0, preserve_time=False):
    """Copy a directory from one filesystem to another, depending on a condition.

    Arguments:
        src_fs (FS or str): Source filesystem (instance or URL).
        src_path (str): Path to a directory on the source filesystem.
        dst_fs (FS or str): Destination filesystem (instance or URL).
        dst_path (str): Path to a directory on the destination filesystem.
        condition (str): Name of the condition to check for each file.
        walker (~fs.walk.Walker, optional): A walker object that will be
            used to scan for files in ``src_fs``. Set this if you only want
            to consider a sub-set of the resources in ``src_fs``.
        on_copy (callable):A function callback called after a single file copy
            is executed. Expected signature is ``(src_fs, src_path, dst_fs,
            dst_path)``.
        workers (int): Use ``worker`` threads to copy data, or ``0`` (default) for
            a single-threaded copy.
        preserve_time (bool): If `True`, try to preserve mtime of the
            resources (defaults to `False`).

    See Also:
        `~fs.copy.copy_file_if` for the full list of supported values for the
        ``condition`` argument.

    """
    on_copy = on_copy or (lambda *args: None)
    walker = walker or Walker()
    _src_path = abspath(normpath(src_path))
    _dst_path = abspath(normpath(dst_path))
    from ._bulk import Copier
    copy_structure(src_fs, dst_fs, walker, src_path, dst_path)
    with manage_fs(src_fs, writeable=False) as _src_fs, manage_fs(dst_fs, create=True) as _dst_fs:
        with _src_fs.lock(), _dst_fs.lock():
            _thread_safe = is_thread_safe(_src_fs, _dst_fs)
            with Copier(num_workers=workers if _thread_safe else 0, preserve_time=preserve_time) as copier:
                for dir_path in walker.files(_src_fs, _src_path):
                    copy_path = combine(_dst_path, frombase(_src_path, dir_path))
                    if _copy_is_necessary(_src_fs, dir_path, _dst_fs, copy_path, condition):
                        copier.copy(_src_fs, dir_path, _dst_fs, copy_path)
                        on_copy(_src_fs, dir_path, _dst_fs, copy_path)
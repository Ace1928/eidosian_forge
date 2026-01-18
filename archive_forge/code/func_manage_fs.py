from __future__ import absolute_import, print_function, unicode_literals
import typing
import collections
import contextlib
import pkg_resources
from ..errors import ResourceReadOnly
from .base import Opener
from .errors import EntryPointError, UnsupportedProtocol
from .parse import parse_fs_url
@contextlib.contextmanager
def manage_fs(self, fs_url, create=False, writeable=False, cwd='.'):
    """Get a context manager to open and close a filesystem.

        Arguments:
            fs_url (FS or str): A filesystem instance or a FS URL.
            create (bool, optional): If `True`, then create the filesystem if
                it doesn't already exist.
            writeable (bool, optional): If `True`, then the filesystem
                must be writeable.
            cwd (str): The current working directory, if opening a
                `~fs.osfs.OSFS`.

        Sometimes it is convenient to be able to pass either a FS object
        *or* an FS URL to a function. This context manager handles the
        required logic for that.

        Example:
            The `~Registry.manage_fs` method can be used to define a small
            utility function::

                >>> def print_ls(list_fs):
                ...     '''List a directory.'''
                ...     with manage_fs(list_fs) as fs:
                ...         print(' '.join(fs.listdir()))

            This function may be used in two ways. You may either pass
            a ``str``, as follows::

                >>> print_list('zip://projects.zip')

            Or, an filesystem instance::

                >>> from fs.osfs import OSFS
                >>> projects_fs = OSFS('~/')
                >>> print_list(projects_fs)

        """
    from ..base import FS

    def assert_writeable(fs):
        if fs.getmeta().get('read_only', True):
            raise ResourceReadOnly(path='/')
    if isinstance(fs_url, FS):
        if writeable:
            assert_writeable(fs_url)
        yield fs_url
    else:
        _fs = self.open_fs(fs_url, create=create, writeable=writeable, cwd=cwd)
        if writeable:
            assert_writeable(_fs)
        try:
            yield _fs
        finally:
            _fs.close()
import os
import os.path
import posixpath
import stat
from .compat import Collection, Iterable, string_types, unicode
def is_symlink(self):
    """
		Returns whether the entry is a symbolic link (:class:`bool`).
		"""
    return stat.S_ISLNK(self._lstat.st_mode)
import os
import os.path
import posixpath
import stat
from .compat import Collection, Iterable, string_types, unicode
def is_file(self, follow_links=None):
    """
		Get whether the entry is a regular file.

		*follow_links* (:class:`bool` or :data:`None`) is whether to follow
		symbolic links. If this is :data:`True`, a symlink to a regular file
		will result in :data:`True`. Default is :data:`None` for :data:`True`.

		Returns whether the entry is a regular file (:class:`bool`).
		"""
    if follow_links is None:
        follow_links = True
    node_stat = self._stat if follow_links else self._lstat
    return stat.S_ISREG(node_stat.st_mode)
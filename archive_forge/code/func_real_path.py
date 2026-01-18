import os
import os.path
import posixpath
import stat
from .compat import Collection, Iterable, string_types, unicode
@property
def real_path(self):
    """
		*real_path* (:class:`str`) is the real path that recursion was
		encountered on.
		"""
    return self.args[0]
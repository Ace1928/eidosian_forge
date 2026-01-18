import os
import os.path
import posixpath
import stat
from .compat import Collection, Iterable, string_types, unicode
@property
def second_path(self):
    """
		*second_path* (:class:`str`) is the second path encountered for
		:attr:`self.real_path <RecursionError.real_path>`.
		"""
    return self.args[2]
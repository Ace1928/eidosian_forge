import os
import os.path
import posixpath
import stat
from .compat import Collection, Iterable, string_types, unicode
def lookup_pattern(name):
    """
	Lookups a registered pattern factory by name.

	*name* (:class:`str`) is the name of the pattern factory.

	Returns the registered pattern factory (:class:`~collections.abc.Callable`).
	If no pattern factory is registered, raises :exc:`KeyError`.
	"""
    return _registered_patterns[name]
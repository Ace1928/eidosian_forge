import os
import os.path
import posixpath
import stat
from .compat import Collection, Iterable, string_types, unicode
def register_pattern(name, pattern_factory, override=None):
    """
	Registers the specified pattern factory.

	*name* (:class:`str`) is the name to register the pattern factory
	under.

	*pattern_factory* (:class:`~collections.abc.Callable`) is used to
	compile patterns. It must accept an uncompiled pattern (:class:`str`)
	and return the compiled pattern (:class:`.Pattern`).

	*override* (:class:`bool` or :data:`None`) optionally is whether to
	allow overriding an already registered pattern under the same name
	(:data:`True`), instead of raising an :exc:`AlreadyRegisteredError`
	(:data:`False`). Default is :data:`None` for :data:`False`.
	"""
    if not isinstance(name, string_types):
        raise TypeError('name:{!r} is not a string.'.format(name))
    if not callable(pattern_factory):
        raise TypeError('pattern_factory:{!r} is not callable.'.format(pattern_factory))
    if name in _registered_patterns and (not override):
        raise AlreadyRegisteredError(name, _registered_patterns[name])
    _registered_patterns[name] = pattern_factory
import json
import warnings
from .base import string_types
def get_register_func(base_class, nickname):
    """Get registrator function.

    Parameters
    ----------
    base_class : type
        base class for classes that will be reigstered
    nickname : str
        nickname of base_class for logging

    Returns
    -------
    a registrator function
    """
    if base_class not in _REGISTRY:
        _REGISTRY[base_class] = {}
    registry = _REGISTRY[base_class]

    def register(klass, name=None):
        """Register functions"""
        assert issubclass(klass, base_class), 'Can only register subclass of %s' % base_class.__name__
        if name is None:
            name = klass.__name__
        name = name.lower()
        if name in registry:
            warnings.warn('\x1b[91mNew %s %s.%s registered with name %s isoverriding existing %s %s.%s\x1b[0m' % (nickname, klass.__module__, klass.__name__, name, nickname, registry[name].__module__, registry[name].__name__), UserWarning, stacklevel=2)
        registry[name] = klass
        return klass
    register.__doc__ = 'Register %s to the %s factory' % (nickname, nickname)
    return register
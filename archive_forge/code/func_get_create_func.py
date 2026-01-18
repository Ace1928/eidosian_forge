import json
import warnings
from .base import string_types
def get_create_func(base_class, nickname):
    """Get creator function

    Parameters
    ----------
    base_class : type
        base class for classes that will be reigstered
    nickname : str
        nickname of base_class for logging

    Returns
    -------
    a creator function
    """
    if base_class not in _REGISTRY:
        _REGISTRY[base_class] = {}
    registry = _REGISTRY[base_class]

    def create(*args, **kwargs):
        """Create instance from config"""
        if len(args):
            name = args[0]
            args = args[1:]
        else:
            name = kwargs.pop(nickname)
        if isinstance(name, base_class):
            assert len(args) == 0 and len(kwargs) == 0, '%s is already an instance. Additional arguments are invalid' % nickname
            return name
        if isinstance(name, dict):
            return create(**name)
        assert isinstance(name, string_types), '%s must be of string type' % nickname
        if name.startswith('['):
            assert not args and (not kwargs)
            name, kwargs = json.loads(name)
            return create(name, **kwargs)
        elif name.startswith('{'):
            assert not args and (not kwargs)
            kwargs = json.loads(name)
            return create(**kwargs)
        name = name.lower()
        assert name in registry, '%s is not registered. Please register with %s.register first' % (str(name), nickname)
        return registry[name](*args, **kwargs)
    create.__doc__ = 'Create a %s instance from config.\n\nParameters\n----------\n%s : str or %s instance\n    class name of desired instance. If is a instance,\n    it will be returned directly.\n**kwargs : dict\n    arguments to be passed to constructor' % (nickname, nickname, base_class.__name__)
    return create
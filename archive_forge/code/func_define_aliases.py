import functools
import itertools
import re
import sys
import warnings
from .deprecation import (
def define_aliases(alias_d, cls=None):
    """
    Class decorator for defining property aliases.

    Use as ::

        @_api.define_aliases({"property": ["alias", ...], ...})
        class C: ...

    For each property, if the corresponding ``get_property`` is defined in the
    class so far, an alias named ``get_alias`` will be defined; the same will
    be done for setters.  If neither the getter nor the setter exists, an
    exception will be raised.

    The alias map is stored as the ``_alias_map`` attribute on the class and
    can be used by `.normalize_kwargs` (which assumes that higher priority
    aliases come last).
    """
    if cls is None:
        return functools.partial(define_aliases, alias_d)

    def make_alias(name):

        @functools.wraps(getattr(cls, name))
        def method(self, *args, **kwargs):
            return getattr(self, name)(*args, **kwargs)
        return method
    for prop, aliases in alias_d.items():
        exists = False
        for prefix in ['get_', 'set_']:
            if prefix + prop in vars(cls):
                exists = True
                for alias in aliases:
                    method = make_alias(prefix + prop)
                    method.__name__ = prefix + alias
                    method.__doc__ = f'Alias for `{prefix + prop}`.'
                    setattr(cls, prefix + alias, method)
        if not exists:
            raise ValueError(f'Neither getter nor setter exists for {prop!r}')

    def get_aliased_and_aliases(d):
        return {*d, *(alias for aliases in d.values() for alias in aliases)}
    preexisting_aliases = getattr(cls, '_alias_map', {})
    conflicting = get_aliased_and_aliases(preexisting_aliases) & get_aliased_and_aliases(alias_d)
    if conflicting:
        raise NotImplementedError(f'Parent class already defines conflicting aliases: {conflicting}')
    cls._alias_map = {**preexisting_aliases, **alias_d}
    return cls
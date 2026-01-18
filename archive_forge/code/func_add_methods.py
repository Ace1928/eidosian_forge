from __future__ import absolute_import
import functools
def add_methods(source_class, blacklist=()):
    """Add wrapped versions of the `api` member's methods to the class.

    Any methods passed in `blacklist` are not added.
    Additionally, any methods explicitly defined on the wrapped class are
    not added.
    """

    def wrap(wrapped_fx, lookup_fx):
        """Wrap a GAPIC method; preserve its name and docstring."""
        if isinstance(lookup_fx, (classmethod, staticmethod)):
            fx = lambda *a, **kw: wrapped_fx(*a, **kw)
            return staticmethod(functools.wraps(wrapped_fx)(fx))
        else:
            fx = lambda self, *a, **kw: wrapped_fx(self.api, *a, **kw)
            return functools.wraps(wrapped_fx)(fx)

    def actual_decorator(cls):
        for name in dir(source_class):
            if name.startswith('_'):
                continue
            if name in blacklist:
                continue
            attr = getattr(source_class, name)
            if not callable(attr):
                continue
            lookup_fx = source_class.__dict__[name]
            fx = wrap(attr, lookup_fx)
            setattr(cls, name, fx)
        return cls
    return actual_decorator
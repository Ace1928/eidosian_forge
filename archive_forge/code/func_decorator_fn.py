import re
from tensorflow.python.types import internal
def decorator_fn(cls):
    if not (isinstance(cls, type) and issubclass(cls, internal.TypeSpec)):
        raise TypeError('Expected `cls` to be a TypeSpec; got %r' % (cls,))
    if cls in _TYPE_SPEC_TO_NAME:
        raise ValueError('Class %s.%s has already been registered with name %s.' % (cls.__module__, cls.__name__, _TYPE_SPEC_TO_NAME[cls]))
    if name in _NAME_TO_TYPE_SPEC:
        raise ValueError('Name %s has already been registered for class %s.%s.' % (name, _NAME_TO_TYPE_SPEC[name].__module__, _NAME_TO_TYPE_SPEC[name].__name__))
    _TYPE_SPEC_TO_NAME[cls] = name
    _NAME_TO_TYPE_SPEC[name] = cls
    return cls
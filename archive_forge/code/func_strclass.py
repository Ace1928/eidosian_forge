from collections import namedtuple, Counter
from os.path import commonprefix
def strclass(cls):
    return '%s.%s' % (cls.__module__, cls.__qualname__)
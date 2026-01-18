from __future__ import print_function, unicode_literals
import typing
import functools
import six
from six import text_type
@classmethod
def catch_all(cls, func):

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except cls:
            raise
        except Exception as e:
            raise cls(exc=e)
    return new_func
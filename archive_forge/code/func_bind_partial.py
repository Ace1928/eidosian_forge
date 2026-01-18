from __future__ import absolute_import, division, print_function
import itertools
import functools
import re
import types
from funcsigs.version import __version__
def bind_partial(self, *args, **kwargs):
    """Get a BoundArguments object, that partially maps the
        passed `args` and `kwargs` to the function's signature.
        Raises `TypeError` if the passed arguments can not be bound.
        """
    return self._bind(args, kwargs, partial=True)
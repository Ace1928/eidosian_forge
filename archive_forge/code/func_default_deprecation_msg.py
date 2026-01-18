import logging
import functools
import inspect
import itertools
import sys
import textwrap
import types
from pyomo.common.errors import DeveloperError
def default_deprecation_msg(obj, user_msg, version, remove_in):
    """Generate the default deprecation message.

    See deprecated() function for argument details.
    """
    if user_msg is None:
        if inspect.isclass(obj):
            _obj = ' class'
        elif inspect.ismethod(obj):
            _obj = ' method'
        elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
            _obj = ' function'
        else:
            _obj = ''
        _qual = getattr(obj, '__qualname__', '') or ''
        if _qual.endswith('.__init__') or _qual.endswith('.__new__'):
            _obj = f' class ({_qual.rsplit('.', 1)[0]})'
        elif _qual and _obj:
            _obj += f' ({_qual})'
        user_msg = 'This%s has been deprecated and may be removed in a future release.' % (_obj,)
    comment = []
    if version:
        comment.append('deprecated in %s' % (version,))
    if remove_in:
        comment.append('will be removed in (or after) %s' % remove_in)
    if comment:
        return user_msg + '  (%s)' % (', '.join(comment),)
    else:
        return user_msg
import re
import collections
from . import _compat, tools
def attr_list(label=None, kwargs=None, attributes=None):
    """Return assembled DOT attribute list string.

    Sorts ``kwargs`` and ``attributes`` if they are plain dicts (to avoid
    unpredictable order from hash randomization in Python 3 versions).

    >>> attr_list()
    ''

    >>> attr_list('spam spam', kwargs={'eggs': 'eggs', 'ham': 'ham ham'})
    ' [label="spam spam" eggs=eggs ham="ham ham"]'

    >>> attr_list(kwargs={'spam': None, 'eggs': ''})
    ' [eggs=""]'
    """
    content = a_list(label, kwargs, attributes)
    if not content:
        return ''
    return ' [%s]' % content
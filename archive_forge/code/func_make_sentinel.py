import re
import sys
import string
import socket
from socket import AF_INET, AF_INET6
from typing import (
from unicodedata import normalize
from ._socket import inet_pton
from idna import encode as idna_encode, decode as idna_decode
def make_sentinel(name='_MISSING', var_name=''):
    """Creates and returns a new **instance** of a new class, suitable for
    usage as a "sentinel", a kind of singleton often used to indicate
    a value is missing when ``None`` is a valid input.

    Args:
        name: Name of the Sentinel
        var_name: Set this name to the name of the variable in its respective
            module enable pickle-ability.

    >>> make_sentinel(var_name='_MISSING')
    _MISSING

    The most common use cases here in boltons are as default values
    for optional function arguments, partly because of its
    less-confusing appearance in automatically generated
    documentation. Sentinels also function well as placeholders in queues
    and linked lists.

    .. note::

        By design, additional calls to ``make_sentinel`` with the same
        values will not produce equivalent objects.

        >>> make_sentinel('TEST') == make_sentinel('TEST')
        False
        >>> type(make_sentinel('TEST')) == type(make_sentinel('TEST'))
        False
    """

    class Sentinel(object):

        def __init__(self):
            self.name = name
            self.var_name = var_name

        def __repr__(self):
            if self.var_name:
                return self.var_name
            return '%s(%r)' % (self.__class__.__name__, self.name)
        if var_name:

            def __reduce__(self):
                return self.var_name

        def __nonzero__(self):
            return False
        __bool__ = __nonzero__
    return Sentinel()
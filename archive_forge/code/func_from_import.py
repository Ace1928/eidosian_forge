from __future__ import absolute_import, division, print_function
import sys
import logging
import contextlib
import copy
import os
from future.utils import PY2, PY3
def from_import(module_name, *symbol_names, **kwargs):
    """
    Example use:
        >>> HTTPConnection = from_import('http.client', 'HTTPConnection')
        >>> HTTPServer = from_import('http.server', 'HTTPServer')
        >>> urlopen, urlparse = from_import('urllib.request', 'urlopen', 'urlparse')

    Equivalent to this on Py3:

        >>> from module_name import symbol_names[0], symbol_names[1], ...

    and this on Py2:

        >>> from future.moves.module_name import symbol_names[0], ...

    or:

        >>> from future.backports.module_name import symbol_names[0], ...

    except that it also handles dotted module names such as ``http.client``.
    """
    if PY3:
        return __import__(module_name)
    else:
        if 'backport' in kwargs and bool(kwargs['backport']):
            prefix = 'future.backports'
        else:
            prefix = 'future.moves'
        parts = prefix.split('.') + module_name.split('.')
        module = importlib.import_module(prefix + '.' + module_name)
        output = [getattr(module, name) for name in symbol_names]
        if len(output) == 1:
            return output[0]
        else:
            return output
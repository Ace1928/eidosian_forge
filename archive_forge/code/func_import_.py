from __future__ import absolute_import, division, print_function
import sys
import logging
import contextlib
import copy
import os
from future.utils import PY2, PY3
def import_(module_name, backport=False):
    """
    Pass a (potentially dotted) module name of a Python 3 standard library
    module. This function imports the module compatibly on Py2 and Py3 and
    returns the top-level module.

    Example use:
        >>> http = import_('http.client')
        >>> http = import_('http.server')
        >>> urllib = import_('urllib.request')

    Then:
        >>> conn = http.client.HTTPConnection(...)
        >>> response = urllib.request.urlopen('http://mywebsite.com')
        >>> # etc.

    Use as follows:
        >>> package_name = import_(module_name)

    On Py3, equivalent to this:

        >>> import module_name

    On Py2, equivalent to this if backport=False:

        >>> from future.moves import module_name

    or to this if backport=True:

        >>> from future.backports import module_name

    except that it also handles dotted module names such as ``http.client``
    The effect then is like this:

        >>> from future.backports import module
        >>> from future.backports.module import submodule
        >>> module.submodule = submodule

    Note that this would be a SyntaxError in Python:

        >>> from future.backports import http.client

    """
    import importlib
    if PY3:
        return __import__(module_name)
    else:
        if backport:
            prefix = 'future.backports'
        else:
            prefix = 'future.moves'
        parts = prefix.split('.') + module_name.split('.')
        modules = []
        for i, part in enumerate(parts):
            sofar = '.'.join(parts[:i + 1])
            modules.append(importlib.import_module(sofar))
        for i, part in reversed(list(enumerate(parts))):
            if i == 0:
                break
            setattr(modules[i - 1], part, modules[i])
        return modules[2]
from __future__ import unicode_literals
import sys
import datetime
from decimal import Decimal
import os
import logging
import hashlib
import warnings
from . import __author__, __copyright__, __license__, __version__
def extend_element(element, base):
    """ Recursively extend the elemnet if it has an extension base."""
    ' Recursion is needed if the extension base itself extends another element.'
    if isinstance(base, dict):
        for i, kk in enumerate(base):
            if isinstance(base, Struct):
                element.insert(kk, base[kk], i)
                if isinstance(base, Struct) and base.namespaces and kk:
                    element.namespaces[kk] = base.namespaces[kk]
                    element.references[kk] = base.references[kk]
        if base.refers_to:
            extend_element(element, base.refers_to)
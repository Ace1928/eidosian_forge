from __future__ import unicode_literals
import sys
import datetime
import sys
import logging
import warnings
import re
import traceback
from . import __author__, __copyright__, __license__, __version__
from .simplexml import SimpleXMLElement, TYPE_MAP, Date, Decimal
def list_methods(self):
    """Return a list of aregistered operations"""
    return [(method, doc) for method, (function, returns, args, doc) in self.methods.items()]
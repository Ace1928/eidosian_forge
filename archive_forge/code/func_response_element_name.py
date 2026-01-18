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
def response_element_name(self, method):
    return '%sResponse' % method
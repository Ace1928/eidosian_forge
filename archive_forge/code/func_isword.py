from __future__ import unicode_literals
import datetime
import re
import string
import time
import warnings
from calendar import monthrange
from io import StringIO
import six
from six import integer_types, text_type
from decimal import Decimal
from warnings import warn
from .. import relativedelta
from .. import tz
@classmethod
def isword(cls, nextchar):
    """ Whether or not the next character is part of a word """
    return nextchar.isalpha()
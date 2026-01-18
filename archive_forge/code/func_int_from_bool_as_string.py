import collections.abc
import math
import re
import unicodedata
import urllib
from oslo_utils._i18n import _
from oslo_utils import encodeutils
def int_from_bool_as_string(subject):
    """Interpret a string as a boolean and return either 1 or 0.

    Any string value in:

        ('True', 'true', 'On', 'on', '1')

    is interpreted as a boolean True.

    Useful for JSON-decoded stuff and config file parsing
    """
    return int(bool_from_string(subject))
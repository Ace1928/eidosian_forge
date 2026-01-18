from __future__ import unicode_literals
import codecs
from email import message_from_file
import json
import logging
import re
from . import DistlibException, __version__
from .compat import StringIO, string_types, text_type
from .markers import interpret
from .util import extract_by_key, get_extras
from .version import get_scheme, PEP440_VERSION_RE
def are_valid_constraints(value):
    for v in value:
        if not scheme.is_valid_matcher(v.split(';')[0]):
            return False
    return True
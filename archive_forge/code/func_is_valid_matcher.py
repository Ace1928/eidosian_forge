import logging
import re
from .compat import string_types
from .util import parse_requirement
def is_valid_matcher(self, s):
    try:
        self.matcher(s)
        result = True
    except UnsupportedVersionError:
        result = False
    return result
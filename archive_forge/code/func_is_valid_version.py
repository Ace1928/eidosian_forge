import logging
import re
from .compat import string_types
from .util import parse_requirement
def is_valid_version(self, s):
    try:
        self.matcher.version_class(s)
        result = True
    except UnsupportedVersionError:
        result = False
    return result
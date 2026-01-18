import logging
import re
from .compat import string_types
from .util import parse_requirement
@property
def exact_version(self):
    result = None
    if len(self._parts) == 1 and self._parts[0][0] in ('==', '==='):
        result = self._parts[0][1]
    return result
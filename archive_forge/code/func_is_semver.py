import logging
import re
from .compat import string_types
from .util import parse_requirement
def is_semver(s):
    return _SEMVER_RE.match(s)
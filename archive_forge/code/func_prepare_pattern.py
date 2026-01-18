from io import StringIO, BytesIO
import codecs
import os
import sys
import re
import errno
from .exceptions import ExceptionPexpect, EOF, TIMEOUT
from .expect import Expecter, searcher_string, searcher_re
def prepare_pattern(pattern):
    if pattern in (TIMEOUT, EOF):
        return pattern
    if isinstance(pattern, self.allowed_string_types):
        return self._coerce_expect_string(pattern)
    self._pattern_type_err(pattern)
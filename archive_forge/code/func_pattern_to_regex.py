from __future__ import unicode_literals
import re
import warnings
from .. import util
from ..compat import unicode
from ..pattern import RegexPattern
@classmethod
def pattern_to_regex(cls, *args, **kw):
    """
		Warn about deprecation.
		"""
    cls._deprecated()
    return super(GitIgnorePattern, cls).pattern_to_regex(*args, **kw)
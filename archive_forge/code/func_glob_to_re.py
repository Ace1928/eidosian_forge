import os, re
import fnmatch
import functools
from distutils.util import convert_path
from distutils.errors import DistutilsTemplateError, DistutilsInternalError
from distutils import log
def glob_to_re(pattern):
    """Translate a shell-like glob pattern to a regular expression; return
    a string containing the regex.  Differs from 'fnmatch.translate()' in
    that '*' does not match "special characters" (which are
    platform-specific).
    """
    pattern_re = fnmatch.translate(pattern)
    sep = os.sep
    if os.sep == '\\':
        sep = '\\\\\\\\'
    escaped = '\\1[^%s]' % sep
    pattern_re = re.sub('((?<!\\\\)(\\\\\\\\)*)\\.', escaped, pattern_re)
    return pattern_re
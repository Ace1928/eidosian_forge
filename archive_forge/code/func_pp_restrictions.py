import collections.abc
import datetime
import email.utils
import functools
import logging
import io
import re
import subprocess
import warnings
import chardet
from debian._util import (
from debian.deprecation import function_deprecated_by
import debian.debian_support
import debian.changelog
def pp_restrictions(restrictions):
    s = []
    for term in restrictions:
        s.append('%s%s' % ('' if term.enabled else '!', term.profile))
    return '<%s>' % ' '.join(s)
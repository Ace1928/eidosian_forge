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
def get_build_date(self):
    if 'build-date' not in self:
        raise KeyError("'Build-Date' field not found in buildinfo")
    timearray = email.utils.parsedate_tz(self['build-date'])
    if timearray is None:
        raise ValueError("Invalid 'Build-Date' field specified")
    ts = email.utils.mktime_tz(timearray)
    return datetime.datetime.fromtimestamp(ts)
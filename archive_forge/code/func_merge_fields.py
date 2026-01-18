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
def merge_fields(self, key, d1, d2=None):
    if d2 is None:
        x1 = self
        x2 = d1
    else:
        x1 = d1
        x2 = d2
    if key in x1 and key in x2:
        merged = self._merge_fields(x1[key], x2[key])
    elif key in x1:
        merged = x1[key]
    elif key in x2:
        merged = x2[key]
    else:
        raise KeyError
    if d2 is None:
        self[key] = merged
        return None
    return merged
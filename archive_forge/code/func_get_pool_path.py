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
def get_pool_path(self):
    """Return the path in the pool where the files would be installed"""
    s = self['files'][0]['section']
    try:
        section, _ = s.split('/')
    except ValueError:
        section = 'main'
    if self['source'].startswith('lib'):
        subdir = self['source'][:4]
    else:
        subdir = self['source'][0]
    return 'pool/%s/%s/%s' % (section, subdir, self['source'])
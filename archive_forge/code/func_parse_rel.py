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
def parse_rel(raw):
    match = cls.__dep_RE.match(raw)
    if match:
        parts = match.groupdict()
        d = {'name': parts['name'], 'archqual': parts['archqual'], 'version': None, 'arch': None, 'restrictions': None}
        if parts['relop'] or parts['version']:
            d['version'] = (parts['relop'], parts['version'])
        if parts['archs']:
            d['arch'] = parse_archs(parts['archs'])
        if parts['restrictions']:
            d['restrictions'] = parse_restrictions(parts['restrictions'])
        return d
    logger.warning('cannot parse package relationship "%s", returning it raw', raw)
    return {'name': raw, 'archqual': None, 'version': None, 'arch': None, 'restrictions': None}
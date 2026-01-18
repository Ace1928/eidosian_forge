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
def parse_restrictions(raw):
    """ split a restriction formula into a list of restriction lists

            Each term in the restriction list is a namedtuple of form:

                (enabled, label)

            where
                enabled: bool: whether the restriction is positive or negative
                profile: the profile name of the term e.g. 'stage1'
            """
    restrictions = []
    groups = cls.__restriction_sep_RE.split(raw.lower().strip('<> '))
    for rgrp in groups:
        group = []
        for restriction in cls.__blank_sep_RE.split(rgrp):
            match = cls.__restriction_RE.match(restriction)
            if match:
                parts = match.groupdict()
                group.append(cls.BuildRestriction(parts['enabled'] != '!', parts['profile']))
        restrictions.append(group)
    return restrictions
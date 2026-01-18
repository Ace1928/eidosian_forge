from __future__ import absolute_import, division, print_function
import abc
import datetime
import errno
import hashlib
import os
import re
from ansible.module_utils import six
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible_collections.community.crypto.plugins.module_utils.crypto.pem import (
from .basic import (
def convert_relative_to_datetime(relative_time_string):
    """Get a datetime.datetime or None from a string in the time format described in sshd_config(5)"""
    parsed_result = re.match('^(?P<prefix>[+-])((?P<weeks>\\d+)[wW])?((?P<days>\\d+)[dD])?((?P<hours>\\d+)[hH])?((?P<minutes>\\d+)[mM])?((?P<seconds>\\d+)[sS]?)?$', relative_time_string)
    if parsed_result is None or len(relative_time_string) == 1:
        return None
    offset = datetime.timedelta(0)
    if parsed_result.group('weeks') is not None:
        offset += datetime.timedelta(weeks=int(parsed_result.group('weeks')))
    if parsed_result.group('days') is not None:
        offset += datetime.timedelta(days=int(parsed_result.group('days')))
    if parsed_result.group('hours') is not None:
        offset += datetime.timedelta(hours=int(parsed_result.group('hours')))
    if parsed_result.group('minutes') is not None:
        offset += datetime.timedelta(minutes=int(parsed_result.group('minutes')))
    if parsed_result.group('seconds') is not None:
        offset += datetime.timedelta(seconds=int(parsed_result.group('seconds')))
    if parsed_result.group('prefix') == '+':
        return datetime.datetime.utcnow() + offset
    else:
        return datetime.datetime.utcnow() - offset
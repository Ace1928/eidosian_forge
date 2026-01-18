from __future__ import absolute_import, division, print_function
import atexit
import ansible.module_utils.common._collections_compat as collections_compat
import json
import os
import re
import socket
import ssl
import hashlib
import time
import traceback
import datetime
from collections import OrderedDict
from ansible.module_utils.compat.version import StrictVersion
from random import randint
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.six import integer_types, iteritems, string_types, raise_from
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.six.moves.urllib.parse import unquote
def ansible_date_time_facts(timestamp):
    date_time_facts = {}
    if timestamp is None:
        return date_time_facts
    utctimestamp = timestamp.astimezone(datetime.timezone.utc)
    date_time_facts['year'] = timestamp.strftime('%Y')
    date_time_facts['month'] = timestamp.strftime('%m')
    date_time_facts['weekday'] = timestamp.strftime('%A')
    date_time_facts['weekday_number'] = timestamp.strftime('%w')
    date_time_facts['weeknumber'] = timestamp.strftime('%W')
    date_time_facts['day'] = timestamp.strftime('%d')
    date_time_facts['hour'] = timestamp.strftime('%H')
    date_time_facts['minute'] = timestamp.strftime('%M')
    date_time_facts['second'] = timestamp.strftime('%S')
    date_time_facts['epoch'] = timestamp.strftime('%s')
    date_time_facts['date'] = timestamp.strftime('%Y-%m-%d')
    date_time_facts['time'] = timestamp.strftime('%H:%M:%S')
    date_time_facts['iso8601_micro'] = utctimestamp.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    date_time_facts['iso8601'] = utctimestamp.strftime('%Y-%m-%dT%H:%M:%SZ')
    date_time_facts['iso8601_basic'] = timestamp.strftime('%Y%m%dT%H%M%S%f')
    date_time_facts['iso8601_basic_short'] = timestamp.strftime('%Y%m%dT%H%M%S')
    date_time_facts['tz'] = timestamp.strftime('%Z')
    date_time_facts['tz_offset'] = timestamp.strftime('%z')
    return date_time_facts
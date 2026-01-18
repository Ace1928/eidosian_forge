from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import raise_from
from ansible.plugins.lookup import LookupBase
from ansible.errors import AnsibleError
from datetime import datetime
@staticmethod
def parse_date_time(date_string):
    try:
        return datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        return datetime.strptime(date_string, '%Y-%m-%d')
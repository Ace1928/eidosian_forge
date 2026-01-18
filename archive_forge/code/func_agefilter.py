from __future__ import absolute_import, division, print_function
import errno
import fnmatch
import grp
import os
import pwd
import re
import stat
import time
from ansible.module_utils.common.text.converters import to_text, to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
def agefilter(st, now, age, timestamp):
    """filter files older than age"""
    if age is None:
        return True
    elif age >= 0 and now - getattr(st, 'st_%s' % timestamp) >= abs(age):
        return True
    elif age < 0 and now - getattr(st, 'st_%s' % timestamp) <= abs(age):
        return True
    return False
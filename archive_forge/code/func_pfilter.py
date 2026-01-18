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
def pfilter(f, patterns=None, excludes=None, use_regex=False):
    """filter using glob patterns"""
    if not patterns and (not excludes):
        return True
    if use_regex:
        if patterns and (not excludes):
            for p in patterns:
                r = re.compile(p)
                if r.match(f):
                    return True
        elif patterns and excludes:
            for p in patterns:
                r = re.compile(p)
                if r.match(f):
                    for e in excludes:
                        r = re.compile(e)
                        if r.match(f):
                            return False
                    return True
    elif patterns and (not excludes):
        for p in patterns:
            if fnmatch.fnmatch(f, p):
                return True
    elif patterns and excludes:
        for p in patterns:
            if fnmatch.fnmatch(f, p):
                for e in excludes:
                    if fnmatch.fnmatch(f, e):
                        return False
                return True
    return False
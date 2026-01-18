from __future__ import (absolute_import, division, print_function)
import string
import json
import re
from ansible.module_utils.six import iteritems
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
def is_hash(password):
    ishash = False
    if len(password) == 41 and password[0] == '*':
        if frozenset(password[1:]).issubset(string.hexdigits):
            ishash = True
    return ishash
from __future__ import absolute_import, division, print_function
import re
from operator import itemgetter
from ansible.module_utils.basic import AnsibleModule
def filter_line_that_match_func(match_func, content):
    return ''.join([line for line in content.splitlines(True) if match_func(line) is not None])
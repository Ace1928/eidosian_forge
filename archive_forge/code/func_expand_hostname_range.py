from __future__ import (absolute_import, division, print_function)
import hashlib
import os
import string
from collections.abc import Mapping
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.inventory.group import to_safe_group_name as original_safe
from ansible.parsing.utils.addresses import parse_address
from ansible.plugins import AnsiblePlugin
from ansible.plugins.cache import CachePluginAdjudicator as CacheObject
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import string_types
from ansible.template import Templar
from ansible.utils.display import Display
from ansible.utils.vars import combine_vars, load_extra_vars
def expand_hostname_range(line=None):
    """
    A helper function that expands a given line that contains a pattern
    specified in top docstring, and returns a list that consists of the
    expanded version.

    The '[' and ']' characters are used to maintain the pseudo-code
    appearance. They are replaced in this function with '|' to ease
    string splitting.

    References: https://docs.ansible.com/ansible/latest/user_guide/intro_inventory.html#hosts-and-groups
    """
    all_hosts = []
    if line:
        head, nrange, tail = line.replace('[', '|', 1).replace(']', '|', 1).split('|')
        bounds = nrange.split(':')
        if len(bounds) != 2 and len(bounds) != 3:
            raise AnsibleError('host range must be begin:end or begin:end:step')
        beg = bounds[0]
        end = bounds[1]
        if len(bounds) == 2:
            step = 1
        else:
            step = bounds[2]
        if not beg:
            beg = '0'
        if not end:
            raise AnsibleError('host range must specify end value')
        if beg[0] == '0' and len(beg) > 1:
            rlen = len(beg)
            if rlen != len(end):
                raise AnsibleError('host range must specify equal-length begin and end formats')

            def fill(x):
                return str(x).zfill(rlen)
        else:
            fill = str
        try:
            i_beg = string.ascii_letters.index(beg)
            i_end = string.ascii_letters.index(end)
            if i_beg > i_end:
                raise AnsibleError('host range must have begin <= end')
            seq = list(string.ascii_letters[i_beg:i_end + 1:int(step)])
        except ValueError:
            seq = range(int(beg), int(end) + 1, int(step))
        for rseq in seq:
            hname = ''.join((head, fill(rseq), tail))
            if detect_range(hname):
                all_hosts.extend(expand_hostname_range(hname))
            else:
                all_hosts.append(hname)
        return all_hosts
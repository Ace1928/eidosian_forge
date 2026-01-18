from __future__ import absolute_import, division, print_function
import json
import re
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.common._collections_compat import Mapping
from ansible.module_utils.connection import Connection, ConnectionError
from ansible.module_utils.six import PY2, PY3
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
def pattern_match_existing(self, output, k):
    """Pattern matching helper for `get_existing`.
        `k` is the command name string. Use the pattern from cmd_ref to
        find a matching string in the output.
        Return regex match object or None.
        """
    ref = self._ref
    pattern = re.compile(ref[k]['getval'])
    multiple = 'multiple' in ref[k].keys()
    match_lines = [re.search(pattern, line) for line in output]
    if 'dict' == ref[k]['kind']:
        match = [m for m in match_lines if m]
        if not match:
            return None
        if len(match) > 1 and (not multiple):
            raise ValueError('get_existing: multiple matches found for property {0}'.format(k))
    else:
        match = [m.groups() for m in match_lines if m]
        if not match:
            return None
        if len(match) > 1 and (not multiple):
            raise ValueError('get_existing: multiple matches found for property {0}'.format(k))
        for item in match:
            index = match.index(item)
            match[index] = list(item)
            if None is match[index][0]:
                match[index].pop(0)
            elif 'no' in match[index][0]:
                match[index].pop(0)
                if not match:
                    return None
    return match
from __future__ import absolute_import, division, print_function
import os
import re
from ansible.errors import AnsibleFilterError
from ansible.module_utils._text import to_native
from ansible.module_utils.common._collections_compat import Mapping
from ansible.module_utils.six import iteritems, string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import Template
def re_finditer(regex, value):
    iter_obj = re.finditer(regex, value)
    values = None
    for each in iter_obj:
        if not values:
            values = each.groupdict()
        else:
            values.update(each.groupdict())
        values['match'] = list(each.groups())
        groups = each.groupdict()
        for group in groups:
            if not values.get('match_all'):
                values['match_all'] = dict()
            if not values['match_all'].get(group):
                values['match_all'][group] = list()
            values['match_all'][group].append(groups[group])
    return values
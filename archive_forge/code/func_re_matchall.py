from __future__ import absolute_import, division, print_function
import os
import re
from ansible.errors import AnsibleFilterError
from ansible.module_utils._text import to_native
from ansible.module_utils.common._collections_compat import Mapping
from ansible.module_utils.six import iteritems, string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import Template
def re_matchall(regex, value):
    objects = list()
    for match in re.findall(regex.pattern, value, re.M):
        obj = {}
        if regex.groupindex:
            for name, index in iteritems(regex.groupindex):
                if len(regex.groupindex) == 1:
                    obj[name] = match
                else:
                    obj[name] = match[index - 1]
            objects.append(obj)
    return objects
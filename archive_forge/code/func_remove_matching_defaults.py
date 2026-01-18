from __future__ import absolute_import, division, print_function
import re
import json
import ast
from copy import copy
from itertools import (count, groupby)
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible.module_utils.common.network import (
from ansible.module_utils.common.validation import check_required_arguments
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def remove_matching_defaults(root, default_entry):
    if isinstance(root, list):
        for list_item in root:
            remove_matching_defaults(list_item, default_entry)
    elif isinstance(root, dict):
        nextobj = root.get(default_entry[0]['name'])
        if nextobj is not None:
            if len(default_entry) > 1:
                remove_matching_defaults(nextobj, default_entry[1:])
            elif nextobj == default_entry[0]['default']:
                root.pop(default_entry[0]['name'])
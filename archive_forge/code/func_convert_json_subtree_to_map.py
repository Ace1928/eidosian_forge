from __future__ import absolute_import, division, print_function
import re
import json
import numbers
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper
from ansible_collections.community.general.plugins.module_utils.snap import snap_runner
def convert_json_subtree_to_map(self, json_subtree, prefix=None):
    option_map = {}
    if not isinstance(json_subtree, dict):
        self.do_raise("Non-dict non-leaf element encountered while parsing option map. The output format of 'snap set' may have changed. Aborting!")
    for key, value in json_subtree.items():
        full_key = key if prefix is None else prefix + '.' + key
        if isinstance(value, (str, float, bool, numbers.Integral)):
            option_map[full_key] = str(value)
        else:
            option_map.update(self.convert_json_subtree_to_map(json_subtree=value, prefix=full_key))
    return option_map
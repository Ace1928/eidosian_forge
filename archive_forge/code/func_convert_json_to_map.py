from __future__ import absolute_import, division, print_function
import re
import json
import numbers
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper
from ansible_collections.community.general.plugins.module_utils.snap import snap_runner
def convert_json_to_map(self, json_string):
    json_object = json.loads(json_string)
    return self.convert_json_subtree_to_map(json_object)
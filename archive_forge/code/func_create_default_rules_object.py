from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.hrobot.plugins.module_utils.robot import (
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.common.text.converters import to_native, to_text
def create_default_rules_object():
    rules = dict()
    for ruleset in RULES:
        rules[ruleset] = []
    return rules
from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.hrobot.plugins.module_utils.robot import (
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.common.text.converters import to_native, to_text
def encode_rule(output, rulename, input):
    for i, rule in enumerate(input['rules'][rulename]):
        for k, v in rule.items():
            if v is not None:
                output['rules[{0}][{1}][{2}]'.format(rulename, i, k)] = v
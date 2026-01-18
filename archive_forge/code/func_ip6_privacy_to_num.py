from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
import re
@staticmethod
def ip6_privacy_to_num(privacy):
    ip6_privacy_values = {'disabled': '0', 'prefer-public-addr': '1 (enabled, prefer public IP)', 'prefer-temp-addr': '2 (enabled, prefer temporary IP)', 'unknown': '-1'}
    if privacy is None:
        return None
    if privacy not in ip6_privacy_values:
        raise AssertionError('{privacy} is invalid ip_privacy6 option'.format(privacy=privacy))
    return ip6_privacy_values[privacy]
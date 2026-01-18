from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
import re
@property
def mac_setting(self):
    if self.type == 'bridge':
        return 'bridge.mac-address'
    else:
        return '802-3-ethernet.cloned-mac-address'
from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
import re
@property
def create_connection_up(self):
    if self.type in ('bond', 'dummy', 'ethernet', 'infiniband', 'wifi'):
        if self.mtu is not None or self.dns4 is not None or self.dns6 is not None:
            return True
    elif self.type == 'team':
        if self.dns4 is not None or self.dns6 is not None:
            return True
    return False
from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
import re
@property
def slave_conn_type(self):
    return self.type in ('ethernet', 'bridge', 'bond', 'vlan', 'team', 'wifi', 'bond-slave', 'bridge-slave', 'team-slave', 'wifi', 'infiniband')
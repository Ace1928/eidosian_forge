from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def call_gluster_cmd(self, *args, **kwargs):
    params = ' '.join((opt for opt in args))
    key_value_pair = ' '.join((' %s %s ' % (key, value) for key, value in kwargs))
    return self._run_command('gluster', ' ' + params + ' ' + key_value_pair)
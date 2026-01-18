from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
def await_online_synchronized_mirror(self):
    self._exec('rabbitmq-upgrade', ['await_online_synchronized_mirror'])
    self.result['changed'] = True
from __future__ import (absolute_import, division, print_function)
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def post_reinstall(self, host):
    wait(service=self._service.service(host.id), condition=lambda h: h.status != hoststate.MAINTENANCE, fail_condition=failed_state, wait=self.param('wait'), timeout=self.param('timeout'))
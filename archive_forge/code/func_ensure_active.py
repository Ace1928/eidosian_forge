from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def ensure_active(self):
    """Wait for the existing Load Balancer to be active"""
    end_time = time.monotonic() + self.wait_timeout
    while time.monotonic() < end_time:
        if self.get_by_id():
            status = self.lb.get('status', None)
            if status is not None:
                if status == 'active':
                    return True
            else:
                self.module.fail_json(msg='Unexpected error; please file a bug: ensure_active')
        else:
            self.module.fail_json(msg='Load Balancer {0} in {1} not found'.format(self.id, self.region))
        time.sleep(10)
    self.module.fail_json(msg='Timed out waiting for Load Balancer {0} in {1} to be active'.format(self.id, self.region))
from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def ensure_power_on(self, droplet_id):
    self.wait_status(droplet_id, ['active', 'off'])
    self.wait_action(droplet_id, {'type': 'power_on'})
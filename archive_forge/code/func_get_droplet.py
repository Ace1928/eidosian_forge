from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def get_droplet(self):
    json_data = self.get_by_id(self.module.params['id'])
    if not json_data and self.unique_name:
        json_data = self.get_by_name(self.module.params['name'])
    return json_data
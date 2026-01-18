from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def get_attached_droplet_ID(self, volume_name, region):
    volume = self.get_block_storage_by_name(volume_name, region)
    if not volume or not volume['droplet_ids']:
        return None
    return volume['droplet_ids'][0]
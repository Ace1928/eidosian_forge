from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def get_block_storage_by_name(self, volume_name, region):
    url = 'volumes?name={0}&region={1}'.format(volume_name, region)
    resp = self.rest.get(url)
    if resp.status_code != 200:
        raise DOBlockStorageException(resp.json['message'])
    volumes = resp.json['volumes']
    if not volumes:
        return None
    return volumes[0]
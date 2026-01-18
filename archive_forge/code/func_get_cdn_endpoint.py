from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def get_cdn_endpoint(self):
    cdns = self.rest.get_paginated_data(base_url='cdn/endpoints?', data_key_name='endpoints')
    found = None
    for cdn in cdns:
        if cdn.get('origin') == self.module.params.get('origin'):
            found = cdn
            for key in ['ttl', 'certificate_id']:
                if self.module.params.get(key) != cdn.get(key):
                    return (found, True)
    return (found, False)
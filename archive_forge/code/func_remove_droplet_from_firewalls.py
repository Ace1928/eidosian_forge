from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def remove_droplet_from_firewalls(self):
    changed = False
    json_data = self.get_droplet()
    if json_data is not None:
        request_params = {}
        droplet = json_data.get('droplet', None)
        droplet_id = droplet.get('id', None)
        request_params['droplet_ids'] = [droplet_id]
        for firewall in self.firewalls:
            if firewall['name'] not in self.module.params['firewall'] and droplet_id in firewall['droplet_ids']:
                response = self.rest.delete('firewalls/{0}/droplets'.format(firewall['id']), data=request_params)
                json_data = response.json
                status_code = response.status_code
                if status_code != 204:
                    err = 'Failed to remove droplet {0} from firewall {1}'.format(droplet_id, firewall['id'])
                    return (err, changed)
                changed = True
    return (None, changed)
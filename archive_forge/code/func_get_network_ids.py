from __future__ import absolute_import, division, print_function
import base64
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def get_network_ids(self, network_names=None):
    if network_names is None:
        network_names = self.module.params.get('networks')
    if not network_names:
        return None
    args = {'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id'), 'zoneid': self.get_zone(key='id'), 'fetch_list': True}
    networks = self.query_api('listNetworks', **args)
    if not networks:
        self.module.fail_json(msg='No networks available')
    network_ids = []
    network_displaytexts = []
    for network_name in network_names:
        for n in networks:
            if network_name in [n['displaytext'], n['name'], n['id']]:
                network_ids.append(n['id'])
                network_displaytexts.append(n['name'])
                break
    if len(network_ids) != len(network_names):
        self.module.fail_json(msg='Could not find all networks, networks list found: %s' % network_displaytexts)
    return network_ids
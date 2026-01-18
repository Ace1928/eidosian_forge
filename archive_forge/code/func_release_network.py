from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, json
from ansible.module_utils.urls import open_url
def release_network(self, network_id='', released_network_name='', released_network_type='lan'):
    """
        Release the network with name 'released_network_name' from the given  supernet network_id
        """
    method = 'get'
    response = None
    if network_id is None or released_network_name is None:
        self.module.exit_json(msg="You must specify those options 'network_id', 'reserved_network_name' and 'reserved_network_size'")
    matched_network_id = ''
    resource_url = 'networks/' + str(network_id) + '/children'
    response = self._get_api_call_ansible_handler(method, resource_url)
    if not response:
        self.module.exit_json(msg=' there is an error in releasing network %r  from network  %s.' % (network_id, released_network_name))
    if response:
        response = json.loads(response)
        for child_net in response:
            if child_net['network'] and child_net['network']['network_name'] == released_network_name:
                matched_network_id = child_net['network']['network_id']
                break
    response = None
    if matched_network_id:
        method = 'delete'
        resource_url = 'networks/' + str(matched_network_id)
        response = self._get_api_call_ansible_handler(method, resource_url, stat_codes=[204])
    else:
        self.module.exit_json(msg=" When release network , could not find the network   %r from the given superent %r' " % (released_network_name, network_id))
    return response
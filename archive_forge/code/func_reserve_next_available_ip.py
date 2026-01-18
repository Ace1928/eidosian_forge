from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, json
from ansible.module_utils.urls import open_url
def reserve_next_available_ip(self, network_id=''):
    """
        Reserve ip address via  Infinity by using rest api
        network_id:  the id of the network that users would like to reserve network from
        return the next available ip address from that given network
        """
    method = 'post'
    resource_url = ''
    response = None
    ip_info = ''
    if not network_id:
        self.module.exit_json(msg="You must specify the option 'network_id'.")
    if network_id:
        resource_url = 'networks/' + str(network_id) + '/reserve_ip'
        response = self._get_api_call_ansible_handler(method, resource_url)
        if response and response.find('[') >= 0 and (response.find(']') >= 0):
            start_pos = response.find('{')
            end_pos = response.find('}')
            ip_info = response[start_pos:end_pos + 1]
    return ip_info
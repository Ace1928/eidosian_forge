from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
def get_network_space_fields(network_space):
    """ Get the network space fields and return as a dict """
    fields = network_space.get_fields(from_cache=True, raw_value=True)
    field_dict = dict(name=fields['name'], network_space_id=fields['id'], netmask=fields['network_config']['netmask'], network=fields['network_config']['network'], default_gateway=fields['network_config']['default_gateway'], interface_ids=fields['interfaces'], service=fields['service'], ips=fields['ips'], properties=fields['properties'], automatic_ip_failback=fields['automatic_ip_failback'], mtu=fields['mtu'], rate_limit=fields['rate_limit'])
    return field_dict
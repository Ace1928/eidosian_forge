from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def set_options_rest(self, parameters):
    """ set attributes for create or modify """

    def add_ip(options, key, value):
        if 'ip' not in options:
            options['ip'] = {}
        options['ip'][key] = value

    def add_location(options, key, value, node=None):
        if 'location' not in options:
            options['location'] = {}
        if key in ['home_node', 'home_port', 'node', 'port', 'broadcast_domain']:
            options['location'][key] = {'name': value}
        else:
            options['location'][key] = value
        if key in ['home_port', 'port']:
            options['location'][key]['node'] = {'name': node}

    def get_node_for_port(parameters, pkey):
        if pkey == 'current_port':
            return parameters.get('current_node') or self.parameters.get('home_node') or self.get_home_node_for_cluster()
        elif pkey == 'home_port':
            return self.parameters.get('home_node') or self.get_home_node_for_cluster()
        else:
            return None
    options, migrate_options, errors = ({}, {}, {})
    create_with_current = False
    if parameters is None:
        parameters = self.parameters
        if self.parameters['interface_type'] == 'fc' and 'home_port' not in self.parameters:
            create_with_current = True
    mapping_params_to_rest = {'admin_status': 'enabled', 'interface_name': 'name', 'vserver': 'svm.name', 'current_port': 'port', 'home_port': 'home_port'}
    if self.parameters['interface_type'] == 'ip':
        mapping_params_to_rest.update({'ipspace': 'ipspace.name', 'service_policy': 'service_policy', 'dns_domain_name': 'dns_zone', 'is_dns_update_enabled': 'ddns_enabled', 'probe_port': 'probe_port', 'subnet_name': 'subnet.name', 'fail_if_subnet_conflicts': 'fail_if_subnet_conflicts', 'address': 'address', 'netmask': 'netmask', 'broadcast_domain': 'broadcast_domain', 'failover_scope': 'failover', 'is_auto_revert': 'auto_revert', 'home_node': 'home_node', 'current_node': 'node'})
    if self.parameters['interface_type'] == 'fc':
        mapping_params_to_rest['data_protocol'] = 'data_protocol'
    ip_keys = ('address', 'netmask')
    location_keys = ('home_port', 'home_node', 'current_port', 'current_node', 'failover_scope', 'is_auto_revert', 'broadcast_domain')
    has_home_port, has_current_port = (False, False)
    if 'home_port' in parameters:
        has_home_port = True
    if 'current_port' in parameters:
        has_current_port = True
    for pkey, rkey in mapping_params_to_rest.items():
        if pkey in parameters:
            if pkey == 'admin_status':
                options[rkey] = parameters[pkey] == 'up'
            elif pkey in ip_keys:
                add_ip(options, rkey, parameters[pkey])
            elif pkey in location_keys:
                if has_home_port and pkey == 'home_node':
                    continue
                if has_current_port and pkey == 'current_node':
                    continue
                dest = migrate_options if rkey in ('node', 'port') and (not create_with_current) else options
                add_location(dest, rkey, parameters[pkey], get_node_for_port(parameters, pkey))
            else:
                options[rkey] = parameters[pkey]
    keys_in_error = ('role', 'failover_group', 'firewall_policy', 'force_subnet_association', 'listen_for_dns_query', 'is_ipv4_link_local')
    for pkey in keys_in_error:
        if pkey in parameters:
            errors[pkey] = parameters[pkey]
    return (options, migrate_options, errors)
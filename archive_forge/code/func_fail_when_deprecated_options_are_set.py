from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
def fail_when_deprecated_options_are_set(self, input_params):
    """ report an error and exit if any deprecated options is set """
    dparms_1g = [x for x in ('ip_address_1g', 'subnet_1g', 'gateway_address_1g', 'mtu_1g', 'bond_mode_1g', 'lacp_1g') if input_params[x] is not None]
    dparms_10g = [x for x in ('ip_address_10g', 'subnet_10g', 'gateway_address_10g', 'mtu_10g', 'bond_mode_10g', 'lacp_10g') if input_params[x] is not None]
    dparms_common = [x for x in ('dns_nameservers', 'dns_search_domains', 'virtual_network_tag') if input_params[x] is not None]
    error_msg = ''
    if dparms_1g and dparms_10g:
        error_msg = 'Please use the new bond_1g and bond_10g options to configure the bond interfaces.'
    elif dparms_1g:
        error_msg = 'Please use the new bond_1g option to configure the bond 1G interface.'
    elif dparms_10g:
        error_msg = 'Please use the new bond_10g option to configure the bond 10G interface.'
    elif dparms_common:
        error_msg = 'Please use the new bond_1g or bond_10g options to configure the bond interfaces.'
    if input_params['method']:
        error_msg = 'This module cannot set or change "method".  ' + error_msg
        dparms_common.append('method')
    if error_msg:
        error_msg += '  The following parameters are deprecated and cannot be used: '
        dparms = dparms_1g
        dparms.extend(dparms_10g)
        dparms.extend(dparms_common)
        error_msg += ', '.join(dparms)
        self.module.fail_json(msg=error_msg)
from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ..module_utils.api import WapiModule
from ..module_utils.api import NIOS_IPV4_NETWORK, NIOS_IPV6_NETWORK
from ..module_utils.api import NIOS_IPV4_NETWORK_CONTAINER, NIOS_IPV6_NETWORK_CONTAINER
from ..module_utils.api import normalize_ib_spec
from ..module_utils.network import validate_ip_address, validate_ip_v6_address
def check_vendor_specific_dhcp_option(module, ib_spec):
    """This function will check if the argument dhcp option belongs to vendor-specific and if yes then will remove
     use_options flag which is not supported with vendor-specific dhcp options.
    """
    for key, value in iteritems(ib_spec):
        if isinstance(module.params[key], list):
            for temp_dict in module.params[key]:
                if 'num' in temp_dict:
                    if temp_dict['num'] in (43, 124, 125, 67, 60):
                        del temp_dict['use_option']
    return ib_spec
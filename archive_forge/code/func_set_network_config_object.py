from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
def set_network_config_object(self, network_params):
    """ set SolidFire network config object """
    network_config = dict()
    if network_params is not None:
        for key in network_params:
            if network_params[key] is not None:
                network_config[key] = network_params[key]
    if network_config:
        return NetworkConfig(**network_config)
    return None
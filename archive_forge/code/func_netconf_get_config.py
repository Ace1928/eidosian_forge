from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec, check_ip_addr
def netconf_get_config(self, **kwargs):
    """ netconf_get_config """
    module = kwargs['module']
    conf_str = kwargs['conf_str']
    xml_str = get_nc_config(module, conf_str)
    return xml_str
from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import execute_nc_action, ce_argument_spec
def netconf_set_action(self, xml_str):
    """ netconf execute action """
    try:
        execute_nc_action(self.network_module, xml_str)
    except TimeoutExpiredError:
        pass
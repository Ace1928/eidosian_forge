from __future__ import (absolute_import, division, print_function)
import copy
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import set_nc_config, get_nc_config
def config_interface_interval_config(self):
    if self.function_lldp_interface_flag == 'intervalINTERFACE':
        tmp = self.get_interface_lldp_disable_pre_config()
        if self.enable_flag == 1 and self.conf_interval_exsit and (tmp[self.ifname] != 'disabled'):
            if self.ifname:
                if self.txinterval:
                    xml_str = CE_NC_MERGE_INTERFACE_INTERVAl_CONFIG % (self.ifname, self.txinterval)
                    ret_xml = set_nc_config(self.module, xml_str)
                    self.check_response(ret_xml, 'INTERFACE_INTERVAL_CONFIG')
                    self.changed = True
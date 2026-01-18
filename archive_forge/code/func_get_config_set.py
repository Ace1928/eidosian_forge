from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.ospf_interfaces.ospf_interfaces import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.rm_templates.ospf_interfaces import (
def get_config_set(self, data):
    """To classify the configurations beased on interface"""
    interface_list = []
    config_set = []
    int_string = ''
    for config_line in data.splitlines():
        ospf_int = re.search('set interfaces \\S+ (\\S+) .*', config_line)
        if ospf_int:
            if ospf_int.group(1) not in interface_list:
                if int_string:
                    config_set.append(int_string)
                interface_list.append(ospf_int.group(1))
                int_string = ''
            int_string = int_string + config_line + '\n'
    if int_string:
        config_set.append(int_string)
    return config_set
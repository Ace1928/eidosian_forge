from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def convert_vlan_ids_range(self, config):
    interface_index = 0
    for conf in config:
        name = conf.get('name', None)
        interface_name = name.replace('/', '%2f')
        mapping_list = conf.get('mapping', [])
        mapping_index = 0
        if mapping_list:
            for mapping in mapping_list:
                vlan_ids = mapping.get('vlan_ids', None)
                if vlan_ids:
                    config[interface_index]['mapping'][mapping_index]['vlan_ids'] = self.vlanIdsRangeStr(vlan_ids)
                mapping_index = mapping_index + 1
        interface_index = interface_index + 1
    return config
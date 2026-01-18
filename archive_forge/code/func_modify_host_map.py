from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_host_map(self, add_host_map, remove_host_map):
    for type, data in sorted(add_host_map.items()):
        if self.use_rest:
            self.add_subsystem_host_map_rest(data, type)
        else:
            self.add_subsystem_host_map(data, type)
    for type, data in sorted(remove_host_map.items()):
        if self.use_rest:
            self.remove_subsystem_host_map_rest(data, type)
        else:
            self.remove_subsystem_host_map(data, type)
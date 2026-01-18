from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_portset_ports(self):
    uuid = None
    if self.use_rest:
        current = self.portset_get_rest()
        if 'uuid' in current:
            uuid = current['uuid']
        current_ports = current['ports'] if 'ports' in current else []
    else:
        current_ports = self.portset_get()
    cd_ports = self.parameters['names']
    if self.parameters['state'] == 'present':
        ports_to_add = [port for port in cd_ports if port not in current_ports]
        if len(ports_to_add) > 0:
            if not self.module.check_mode:
                if self.use_rest:
                    self.add_portset_ports_rest(uuid, ports_to_add)
                else:
                    for port in ports_to_add:
                        self.add_portset_ports(port)
            self.na_helper.changed = True
    if self.parameters['state'] == 'absent':
        ports_to_remove = [port for port in cd_ports if port in current_ports]
        if len(ports_to_remove) > 0:
            if not self.module.check_mode:
                for port in ports_to_remove:
                    self.remove_portset_ports(port, uuid)
            self.na_helper.changed = True
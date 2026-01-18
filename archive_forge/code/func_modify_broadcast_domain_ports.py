from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_broadcast_domain_ports(self):
    """
        compare current and desire ports. Call add or remove ports methods if needed.
        :return: None.
        """
    current_ports = self.get_broadcast_domain_ports()
    cd_ports = self.parameters['names']
    if self.parameters['state'] == 'present':
        ports_to_add = [port for port in cd_ports if port not in current_ports]
        if len(ports_to_add) > 0:
            if not self.module.check_mode:
                if self.use_rest:
                    self.add_broadcast_domain_ports_rest(self.ports_to_add_from_desired(ports_to_add))
                else:
                    self.add_broadcast_domain_ports(ports_to_add)
            self.na_helper.changed = True
    if self.parameters['state'] == 'absent':
        ports_to_remove = [port for port in cd_ports if port in current_ports]
        if len(ports_to_remove) > 0:
            if not self.module.check_mode:
                if self.use_rest:
                    self.remove_broadcast_domain_ports_rest(ports_to_remove, self.parameters['ipspace'])
                else:
                    self.remove_broadcast_domain_ports(ports_to_remove)
            self.na_helper.changed = True
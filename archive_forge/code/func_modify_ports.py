from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_ports(self, current_ports):
    add_ports = set(self.parameters['ports']) - set(current_ports)
    remove_ports = set(current_ports) - set(self.parameters['ports'])
    for port in add_ports:
        self.add_port_to_if_grp(port)
    for port in remove_ports:
        self.remove_port_to_if_grp(port)
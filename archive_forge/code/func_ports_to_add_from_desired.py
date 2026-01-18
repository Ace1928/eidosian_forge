from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def ports_to_add_from_desired(self, ports):
    ports_to_add = []
    for port in ports:
        for port_to_add in self.desired_ports:
            if port == port_to_add['name']:
                ports_to_add.append({'uuid': port_to_add['uuid']})
    return ports_to_add
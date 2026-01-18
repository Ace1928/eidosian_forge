from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def get_node_for_port(parameters, pkey):
    if pkey == 'current_port':
        return parameters.get('current_node') or self.parameters.get('home_node') or self.get_home_node_for_cluster()
    elif pkey == 'home_port':
        return self.parameters.get('home_node') or self.get_home_node_for_cluster()
    else:
        return None
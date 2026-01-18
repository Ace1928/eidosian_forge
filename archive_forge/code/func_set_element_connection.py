from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def set_element_connection(self, kind):
    if kind == 'source':
        elem = netapp_utils.create_sf_connection(module=self.module, host_options=self.parameters['peer_options'])
    elif kind == 'destination':
        elem = netapp_utils.create_sf_connection(module=self.module, host_options=self.parameters)
    elementsw_helper = NaElementSWModule(elem)
    return (elementsw_helper, elem)
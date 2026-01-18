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
def set_source_peer(self):
    if self.parameters.get('source_hostname') is None and self.parameters.get('peer_options') is None:
        if self.parameters.get('connection_type') == 'ontap_elementsw':
            return self.module.fail_json(msg='Error: peer_options are required to identify ONTAP cluster with connection_type: ontap_elementsw')
        if self.parameters.get('connection_type') == 'elementsw_ontap':
            return self.module.fail_json(msg='Error: peer_options are required to identify SolidFire cluster with connection_type: elementsw_ontap')
    if self.parameters.get('source_hostname') is not None:
        self.parameters['peer_options'] = dict(hostname=self.parameters.get('source_hostname'), username=self.parameters.get('source_username'), password=self.parameters.get('source_password'))
    elif self.na_helper.safe_get(self.parameters, ['peer_options', 'hostname']):
        self.parameters['source_hostname'] = self.parameters['peer_options']['hostname']
    if 'peer_options' in self.parameters:
        netapp_utils.setup_host_options_from_module_params(self.parameters['peer_options'], self.module, netapp_utils.na_ontap_host_argument_spec_peer().keys())
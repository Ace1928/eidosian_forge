from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def derive_fc_data_protocol(self):
    protocols = self.parameters.get('protocols')
    if not protocols:
        return
    if len(protocols) > 1:
        self.module.fail_json(msg='A single protocol entry is expected for FC interface, got %s.' % protocols)
    mapping = {'fc-nvme': 'fc_nvme', 'fc_nvme': 'fc_nvme', 'fcp': 'fcp'}
    if protocols[0] not in mapping:
        self.module.fail_json(msg='Unexpected protocol value %s.' % protocols[0])
    data_protocol = mapping[protocols[0]]
    if 'data_protocol' in self.parameters and self.parameters['data_protocol'] != data_protocol:
        self.module.fail_json(msg='Error: mismatch between configured data_protocol: %s and data_protocols: %s' % (self.parameters['data_protocol'], protocols))
    self.parameters['data_protocol'] = data_protocol
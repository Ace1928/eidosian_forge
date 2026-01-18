from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def form_portset_info(self, record):
    self.uuid = record['uuid']
    if self.parameters.get('type') is None:
        self.parameters['type'] = record['protocol']
    portset_info = {'type': record['protocol'], 'ports': []}
    if 'interfaces' in record:
        for lif in record['interfaces']:
            for key, value in lif.items():
                if key in ['fc', 'ip']:
                    self.lifs_info[value['name']] = {'lif_type': key, 'uuid': value['uuid']}
                    portset_info['ports'].append(value['name'])
    return portset_info
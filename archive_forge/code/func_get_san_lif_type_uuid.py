from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_san_lif_type_uuid(self, lif, portset_type):
    api = 'network/%s/interfaces' % portset_type
    query = {'name': lif, 'svm.name': self.parameters['vserver']}
    record, error = rest_generic.get_one_record(self.rest_api, api, query)
    return (record, error)
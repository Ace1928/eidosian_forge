from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_certificate_uuid(self):
    """Get certificate UUID."""
    api = 'security/certificates'
    query = {'name': self.parameters['name']}
    if self.parameters.get('svm'):
        query['svm.name'] = self.parameters['svm']
    else:
        query['scope'] = 'cluster'
    record, error = rest_generic.get_one_record(self.rest_api, api, query, 'uuid')
    if error:
        self.module.fail_json(msg='Error fetching uuid for certificate %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
    if record:
        return record['uuid']
    return None
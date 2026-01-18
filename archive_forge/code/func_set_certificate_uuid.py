from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver, zapis_svm
def set_certificate_uuid(self):
    """Retrieve certicate uuid for 9.8 or later"""
    api = 'security/certificates'
    query = {'name': self.parameters['web']['certificate']['name'], 'svm.name': self.parameters['name'], 'type': 'server'}
    record, error = rest_generic.get_one_record(self.rest_api, api, query)
    if error:
        self.module.fail_json(msg='Error retrieving certificate %s: %s' % (self.parameters['web']['certificate'], error))
    if not record:
        self.module.fail_json(msg='Error certificate not found: %s.  Current certificates with type=server: %s' % (self.parameters['web']['certificate'], self.get_certificates('server')))
    self.parameters['web']['certificate']['uuid'] = record['uuid']
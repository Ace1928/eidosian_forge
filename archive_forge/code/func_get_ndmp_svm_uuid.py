from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
def get_ndmp_svm_uuid(self):
    """
            Get a svm's UUID
            :return: uuid of the node
            """
    params = {'svm.name': self.parameters['vserver']}
    api = 'protocols/ndmp/svms'
    message, error = self.rest_api.get(api, params)
    if error is not None:
        self.module.fail_json(msg=error)
    if 'records' in message and len(message['records']) == 0:
        self.module.fail_json(msg='Error fetching uuid for vserver %s: ' % self.parameters['vserver'])
    if len(message.keys()) == 0:
        error = 'No information collected from %s: %s' % (api, repr(message))
        self.module.fail_json(msg=error)
    elif 'records' not in message:
        error = 'Unexpected response from %s: %s' % (api, repr(message))
        self.module.fail_json(msg=error)
    return message['records'][0]['svm']['uuid']
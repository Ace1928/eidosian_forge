from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def form_create_modify_body(self, params=None):
    """
        Returns body for create or modify.
        """
    if params is None:
        params = self.parameters
    body = {}
    keys = ['name', 'action', 'authentication_method', 'enabled', 'secret_key', 'local_endpoint', 'local_identity', 'remote_identity', 'protocol', 'remote_endpoint']
    for key in keys:
        if key in params:
            body[key] = self.parameters[key]
    if 'certificate' in params:
        body['certificate.name'] = self.parameters['certificate']
    if 'ipspace' in params:
        body['ipspace.name'] = self.parameters['ipspace']
    if 'svm' in params:
        body['svm.name'] = self.parameters['svm']
    return body
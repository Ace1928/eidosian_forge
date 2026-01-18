from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_cifs_server_rest(self, current, modify):
    """
        Modify the state of CIFS server.
        rename: cifs server should be in stopped state
        """
    if not self.use_rest:
        return self.modify_cifs_server()
    body, query = self.create_modify_body_rest(modify)
    api = 'protocols/cifs/services'
    dummy, error = rest_generic.patch_async(self.rest_api, api, current['svm']['uuid'], body, query)
    if error is not None:
        self.module.fail_json(msg='Error on modifying cifs server: %s' % error)
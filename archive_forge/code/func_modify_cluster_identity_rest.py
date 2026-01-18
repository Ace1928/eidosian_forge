from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_cluster_identity_rest(self, modify):
    """
        Modifies the cluster identity
        """
    if 'certificate' in modify:
        self.modify_web_services()
    body = self.create_cluster_body(modify)
    dummy, error = rest_generic.patch_async(self.rest_api, 'cluster', None, body)
    if error:
        self.module.fail_json(msg='Error modifying cluster identity details %s: %s' % (self.parameters['cluster_name'], to_native(error)), exception=traceback.format_exc())
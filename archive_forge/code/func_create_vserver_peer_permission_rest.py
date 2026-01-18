from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_vserver_peer_permission_rest(self):
    """
        Creates an SVM peer permission.
        """
    api = 'svm/peer-permissions'
    body = {'svm.name': self.parameters['vserver'], 'cluster_peer.name': self.parameters['cluster_peer'], 'applications': self.parameters['applications']}
    record, error = rest_generic.post_async(self.rest_api, api, body)
    if error:
        self.module.fail_json(msg='Error on creating vserver peer permissions: %s' % error)
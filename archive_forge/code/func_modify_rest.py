from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def modify_rest(self, modify, uuid):
    body = {}
    if 'banner' in modify:
        body['banner'] = modify['banner']
    if 'motd_message' in modify:
        body['message'] = modify['motd_message']
    if modify.get('show_cluster_motd') is not None:
        body['show_cluster_message'] = modify['show_cluster_motd']
    if body:
        api = 'security/login/messages'
        dummy, error = rest_generic.patch_async(self.rest_api, api, uuid, body)
        if error:
            keys = list(body.keys())
            self.module.fail_json(msg='Error modifying %s: %s' % (', '.join(keys), error))
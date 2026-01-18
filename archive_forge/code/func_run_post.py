from __future__ import absolute_import, division, print_function
import codecs
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_text, to_bytes
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_owning_resource, rest_vserver
def run_post(self, gather_subset_info):
    api = gather_subset_info['api_call']
    post_return, error = self.rest_api.post(api, None)
    if error:
        return None
    dummy, error = self.rest_api.wait_on_job(post_return['job'], increment=5)
    if error:
        self.module.fail_json(msg='%s' % error)
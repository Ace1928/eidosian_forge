from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_spbm import SPBM
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
def create_storage_policy(self, policy, pbm_client, results):
    profile_ids = pbm_client.PbmCreate(createSpec=self.create_mob_pbm_create_spec(self.format_tag_mob_id(self.params.get('tag_category')), None, [self.params.get('tag_name')], self.params.get('tag_category'), self.params.get('description'), self.params.get('name')))
    policy = pbm_client.PbmRetrieveContent(profileIds=[profile_ids])
    self.format_results_and_exit(results, policy[0], True)
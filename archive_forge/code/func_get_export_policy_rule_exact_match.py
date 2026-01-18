from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def get_export_policy_rule_exact_match(self, query):
    """ fetch rules based on attributes
            REST queries only allow for one value at a time in a list, so:
            1. get a short list of matches using a simple query
            2. then look for an exact match
        """
    api = 'protocols/nfs/export-policies/%s/rules' % self.policy_id
    query.update(self.create_query(self.parameters))
    records, error = rest_generic.get_0_or_more_records(self.rest_api, api, query)
    if error:
        if "entry doesn't exist" in error:
            return None
        self.module.fail_json(msg='Error on fetching export policy rules: %s' % error)
    return self.match_export_policy_rule_exactly(records, query, is_rest=True)
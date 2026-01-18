from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def modify_snapmirror_policy_rules_rest(self, uuid, obsolete_rules, unmodified_rules, modified_rules, new_rules):
    api = 'snapmirror/policies'
    if not modified_rules and (not new_rules) and (not obsolete_rules):
        return
    rules = unmodified_rules + modified_rules + new_rules
    body = {'retention': self.create_snapmirror_policy_retention_obj_for_rest(rules)}
    dummy, error = rest_generic.patch_async(self.rest_api, api, uuid, body)
    if error:
        self.module.fail_json(msg='Error modifying snapmirror policy rules: %s' % error)
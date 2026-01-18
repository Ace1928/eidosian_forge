from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def modify_export_policy_rule(self, params, rule_index=None, rename=False):
    """
        Modify an existing export policy rule
        :param params: dict() of attributes with desired values
        :return: None
        """
    if self.use_rest:
        return self.modify_export_policy_rule_rest(params, rule_index, rename)
    params.pop('rule_index', None)
    if params:
        export_rule_modify = netapp_utils.zapi.NaElement.create_node_with_children('export-rule-modify', **{'policy-name': self.parameters['name'], 'rule-index': str(rule_index)})
        self.add_parameters_for_create_or_modify(export_rule_modify, params)
        try:
            self.server.invoke_successfully(export_rule_modify, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error modifying export policy rule index %s: %s' % (rule_index, to_native(error)), exception=traceback.format_exc())
    if rename:
        export_rule_set_index = netapp_utils.zapi.NaElement.create_node_with_children('export-rule-set-index', **{'policy-name': self.parameters['name'], 'rule-index': str(self.parameters['from_rule_index']), 'new-rule-index': str(self.parameters['rule_index'])})
        try:
            self.server.invoke_successfully(export_rule_set_index, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error reindexing export policy rule index %s: %s' % (self.parameters['from_rule_index'], to_native(error)), exception=traceback.format_exc())
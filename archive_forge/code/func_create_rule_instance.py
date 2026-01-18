from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
from ansible.module_utils._text import to_native
from datetime import timedelta
def create_rule_instance(params):
    rule = params.copy()
    rule['metric_resource_uri'] = rule.get('metric_resource_uri', self.target)
    rule['time_grain'] = timedelta(minutes=rule.get('time_grain', 0))
    rule['time_window'] = timedelta(minutes=rule.get('time_window', 0))
    rule['cooldown'] = timedelta(minutes=rule.get('cooldown', 0))
    return ScaleRule(metric_trigger=MetricTrigger(**rule), scale_action=ScaleAction(**rule))
from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils._text import to_native
def rule_to_dict(rule):
    if not rule:
        return dict()
    result = dict(metric_name=to_native(rule.metric_trigger.metric_name), metric_resource_uri=to_native(rule.metric_trigger.metric_resource_uri), time_grain=timedelta_to_minutes(rule.metric_trigger.time_grain), statistic=get_enum_value(rule.metric_trigger.statistic), time_window=timedelta_to_minutes(rule.metric_trigger.time_window), time_aggregation=get_enum_value(rule.metric_trigger.time_aggregation), operator=get_enum_value(rule.metric_trigger.operator), threshold=float(rule.metric_trigger.threshold))
    if rule.scale_action and to_native(rule.scale_action.direction) != 'None':
        result['direction'] = get_enum_value(rule.scale_action.direction)
        result['type'] = get_enum_value(rule.scale_action.type)
        result['value'] = to_native(rule.scale_action.value)
        result['cooldown'] = timedelta_to_minutes(rule.scale_action.cooldown)
    return result
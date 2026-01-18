import datetime
import time
from copy import deepcopy
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.botocore import normalize_boto3_result
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def compare_and_update_configuration(client, module, current_lifecycle_rules, rule):
    purge_transitions = module.params.get('purge_transitions')
    rule_id = module.params.get('rule_id')
    lifecycle_configuration = dict(Rules=[])
    changed = False
    appended = False
    if current_lifecycle_rules:
        for existing_rule in current_lifecycle_rules:
            if rule.get('ID') == existing_rule.get('ID') and rule['Filter'].get('Prefix', '') != existing_rule.get('Filter', {}).get('Prefix', ''):
                existing_rule.pop('ID')
            elif rule_id is None and rule['Filter'].get('Prefix', '') == existing_rule.get('Filter', {}).get('Prefix', ''):
                existing_rule.pop('ID')
            if rule.get('ID') == existing_rule.get('ID'):
                changed_, appended_ = update_or_append_rule(rule, existing_rule, purge_transitions, lifecycle_configuration)
                changed = changed_ or changed
                appended = appended_ or appended
            else:
                lifecycle_configuration['Rules'].append(existing_rule)
        if not appended:
            lifecycle_configuration['Rules'].append(rule)
            changed = True
    else:
        lifecycle_configuration['Rules'].append(rule)
        changed = True
    return (changed, lifecycle_configuration)
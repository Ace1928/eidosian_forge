import json
from datetime import datetime
from typing import Optional
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.backup import get_plan_details
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
def plan_update_needed(existing_plan: dict, new_plan: dict) -> bool:
    """
    Determines whether existing and new plan rules/settings match.

    existing_plan: Dict of existing plan details including rules and advanced settings,
        in snake-case format
    new_plan: Dict of existing plan details including rules and advanced settings, in
        snake-case format
    """
    update_needed = False
    existing_rules = json.dumps([{key: val for key, val in rule.items() if key != 'rule_id'} for rule in existing_plan['backup_plan']['rules']], sort_keys=True)
    new_rules = json.dumps(new_plan['rules'], sort_keys=True)
    if not existing_rules or existing_rules != new_rules:
        update_needed = True
    existing_advanced_backup_settings = json.dumps(existing_plan['backup_plan'].get('advanced_backup_settings', []), sort_keys=True)
    new_advanced_backup_settings = json.dumps(new_plan.get('advanced_backup_settings', []), sort_keys=True)
    if existing_advanced_backup_settings != new_advanced_backup_settings:
        update_needed = True
    return update_needed
import json
from datetime import datetime
from typing import Optional
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.backup import get_plan_details
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
def format_check_mode_response(plan_name: str, plan: dict, tags: dict, delete: bool=False) -> dict:
    """
    Formats plan details in check mode to match result expectations.

    plan_name: Name of backup plan
    plan: Dict of plan details including name, rules, and advanced settings
    tags: Optional dict of plan tags
    delete: Whether the response is for a delete action
    """
    timestamp = datetime.now().isoformat()
    if delete:
        return {'backup_plan_name': plan_name, 'backup_plan_id': '', 'backup_plan_arn': '', 'deletion_date': timestamp, 'version_id': ''}
    else:
        return {'backup_plan_name': plan_name, 'backup_plan_id': '', 'backup_plan_arn': '', 'creation_date': timestamp, 'version_id': '', 'backup_plan': {'backup_plan_name': plan_name, 'rules': plan['rules'], 'advanced_backup_settings': plan['advanced_backup_settings'], 'tags': tags}}
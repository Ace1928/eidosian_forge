import datetime
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def check_user_or_group_update_needed(module, ec2):
    existing_create_vol_permission = _describe_snapshot_attribute(module, ec2, module.params.get('snapshot_id'))
    purge_permission = module.params.get('purge_create_vol_permission')
    supplied_group_names = module.params.get('group_names')
    supplied_user_ids = module.params.get('user_ids')
    if any((item.get('Group') == 'all' for item in existing_create_vol_permission)) and (not purge_permission):
        return False
    if supplied_group_names:
        existing_group_names = {item.get('Group') for item in existing_create_vol_permission or []}
        if set(supplied_group_names) == set(existing_group_names):
            return False
        else:
            return True
    if supplied_user_ids:
        existing_user_ids = {item.get('UserId') for item in existing_create_vol_permission or []}
        if set(supplied_user_ids) == set(existing_user_ids):
            return False
        else:
            return True
    if purge_permission and existing_create_vol_permission == []:
        return False
    return True
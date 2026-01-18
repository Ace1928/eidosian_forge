import datetime
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def create_snapshot_ansible_module():
    argument_spec = dict(volume_id=dict(), description=dict(), instance_id=dict(), snapshot_id=dict(), device_name=dict(), wait=dict(type='bool', default=True), wait_timeout=dict(type='int', default=600), last_snapshot_min_age=dict(type='int', default=0), snapshot_tags=dict(type='dict', default=dict()), state=dict(choices=['absent', 'present'], default='present'), modify_create_vol_permission=dict(type='bool'), purge_create_vol_permission=dict(type='bool', default=False), user_ids=dict(type='list', elements='str'), group_names=dict(type='list', elements='str', choices=['all']))
    mutually_exclusive = [('instance_id', 'snapshot_id', 'volume_id'), ('group_names', 'user_ids')]
    required_if = [('state', 'absent', ('snapshot_id',)), ('purge_create_vol_permission', True, ('modify_create_vol_permission',))]
    required_one_of = [('instance_id', 'snapshot_id', 'volume_id')]
    required_together = [('instance_id', 'device_name')]
    module = AnsibleAWSModule(argument_spec=argument_spec, mutually_exclusive=mutually_exclusive, required_if=required_if, required_one_of=required_one_of, required_together=required_together, supports_check_mode=True)
    return module
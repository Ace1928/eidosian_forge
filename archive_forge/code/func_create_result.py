from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.rds import ensure_tags
from ansible_collections.amazon.aws.plugins.module_utils.rds import get_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
def create_result(changed, subnet_group=None):
    if subnet_group is None:
        return dict(changed=changed)
    result_subnet_group = dict(subnet_group)
    result_subnet_group['name'] = result_subnet_group.get('db_subnet_group_name')
    result_subnet_group['description'] = result_subnet_group.get('db_subnet_group_description')
    result_subnet_group['status'] = result_subnet_group.get('subnet_group_status')
    result_subnet_group['subnet_ids'] = create_subnet_list(subnet_group.get('subnets'))
    return dict(changed=changed, subnet_group=result_subnet_group)
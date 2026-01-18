from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def describe_subnets(connection, module):
    """
    Describe Subnets.

    module  : AnsibleAWSModule object
    connection  : boto3 client connection object
    """
    filters = ansible_dict_to_boto3_filter_list(module.params.get('filters'))
    subnet_ids = module.params.get('subnet_ids')
    if subnet_ids is None:
        subnet_ids = []
    subnet_info = list()
    try:
        response = describe_subnets_with_backoff(connection, subnet_ids, filters)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to describe subnets')
    for subnet in response['Subnets']:
        subnet['id'] = subnet['SubnetId']
        subnet_info.append(camel_dict_to_snake_dict(subnet))
        subnet_info[-1]['tags'] = boto3_tag_list_to_ansible_dict(subnet.get('Tags', []))
    module.exit_json(subnets=subnet_info)
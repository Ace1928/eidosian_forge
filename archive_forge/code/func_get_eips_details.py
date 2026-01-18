from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def get_eips_details(module):
    connection = module.client('ec2', retry_decorator=AWSRetry.jittered_backoff())
    filters = module.params.get('filters')
    try:
        response = connection.describe_addresses(aws_retry=True, Filters=ansible_dict_to_boto3_filter_list(filters))
    except (BotoCoreError, ClientError) as e:
        module.fail_json_aws(e, msg='Error retrieving EIPs')
    addresses = camel_dict_to_snake_dict(response)['addresses']
    for address in addresses:
        if 'tags' in address:
            address['tags'] = boto3_tag_list_to_ansible_dict(address['tags'])
    return addresses
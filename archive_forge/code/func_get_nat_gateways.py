from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.botocore import normalize_boto3_result
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def get_nat_gateways(client, module):
    params = dict()
    nat_gateways = list()
    params['Filter'] = ansible_dict_to_boto3_filter_list(module.params.get('filters'))
    params['NatGatewayIds'] = module.params.get('nat_gateway_ids')
    try:
        result = normalize_boto3_result(_describe_nat_gateways(client, module, **params))
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, 'Unable to describe NAT gateways.')
    for gateway in result:
        converted_gateway = camel_dict_to_snake_dict(gateway)
        if 'tags' in converted_gateway:
            converted_gateway['tags'] = boto3_tag_list_to_ansible_dict(converted_gateway['tags'])
        nat_gateways.append(converted_gateway)
    return nat_gateways
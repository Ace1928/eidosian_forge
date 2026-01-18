from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def list_carrier_gateways(connection, module):
    params = dict()
    params['Filters'] = ansible_dict_to_boto3_filter_list(module.params.get('filters'))
    if module.params.get('carrier_gateway_ids'):
        params['CarrierGatewayIds'] = module.params.get('carrier_gateway_ids')
    try:
        all_carrier_gateways = connection.describe_carrier_gateways(aws_retry=True, **params)
    except is_boto3_error_code('InvalidCarrierGatewayID.NotFound'):
        module.fail_json('CarrierGateway not found')
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, 'Unable to describe carrier gateways')
    return [get_carrier_gateway_info(cagw) for cagw in all_carrier_gateways['CarrierGateways']]
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
@AWSRetry.exponential_backoff()
def describe_transit_gateways(self):
    """
        Describe transit gateways.

        module  : AnsibleAWSModule object
        connection  : boto3 client connection object
        """
    filters = ansible_dict_to_boto3_filter_list(self._module.params['filters'])
    transit_gateway_ids = self._module.params['transit_gateway_ids']
    transit_gateway_info = list()
    try:
        response = self._connection.describe_transit_gateways(TransitGatewayIds=transit_gateway_ids, Filters=filters)
    except is_boto3_error_code('InvalidTransitGatewayID.NotFound'):
        self._results['transit_gateways'] = []
        return
    for transit_gateway in response['TransitGateways']:
        transit_gateway_info.append(camel_dict_to_snake_dict(transit_gateway, ignore_list=['Tags']))
        transit_gateway_info[-1]['tags'] = boto3_tag_list_to_ansible_dict(transit_gateway.get('Tags', []))
    self._results['transit_gateways'] = transit_gateway_info
    return
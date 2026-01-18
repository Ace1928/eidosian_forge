from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def describe_spot_instance_requests(connection, module):
    params = {}
    if module.params.get('filters'):
        params['Filters'] = ansible_dict_to_boto3_filter_list(module.params.get('filters'))
    if module.params.get('spot_instance_request_ids'):
        params['SpotInstanceRequestIds'] = module.params.get('spot_instance_request_ids')
    try:
        describe_spot_instance_requests_response = _describe_spot_instance_requests(connection, **params)['SpotInstanceRequests']
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to describe spot instance requests')
    spot_request = []
    for response_list_item in describe_spot_instance_requests_response:
        spot_request.append(camel_dict_to_snake_dict(response_list_item))
    if len(spot_request) == 0:
        module.exit_json(msg='No spot requests found for specified options')
    module.exit_json(spot_request=spot_request)
import datetime
import json
import traceback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.botocore import normalize_boto3_result
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def setup_removal(client, module):
    params = dict()
    changed = False
    if module.check_mode:
        try:
            exists = client.describe_vpc_endpoints(aws_retry=True, VpcEndpointIds=[module.params.get('vpc_endpoint_id')])
            if exists:
                result = {'msg': 'Would have deleted VPC Endpoint if not in check mode'}
                changed = True
        except is_boto3_error_code('InvalidVpcEndpointId.NotFound'):
            result = {'msg': 'Endpoint does not exist, nothing to delete.'}
            changed = False
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            module.fail_json_aws(e, msg='Failed to get endpoints')
        return (changed, result)
    if isinstance(module.params.get('vpc_endpoint_id'), string_types):
        params['VpcEndpointIds'] = [module.params.get('vpc_endpoint_id')]
    else:
        params['VpcEndpointIds'] = module.params.get('vpc_endpoint_id')
    try:
        result = client.delete_vpc_endpoints(aws_retry=True, **params)['Unsuccessful']
        if len(result) < len(params['VpcEndpointIds']):
            changed = True
        for r in result:
            try:
                raise botocore.exceptions.ClientError(r, 'delete_vpc_endpoints')
            except is_boto3_error_code('InvalidVpcEndpoint.NotFound'):
                continue
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, 'Failed to delete VPC endpoint')
    return (changed, result)
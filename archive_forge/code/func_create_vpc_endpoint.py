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
def create_vpc_endpoint(client, module):
    params = dict()
    changed = False
    token_provided = False
    params['VpcId'] = module.params.get('vpc_id')
    params['VpcEndpointType'] = module.params.get('vpc_endpoint_type')
    params['ServiceName'] = module.params.get('service')
    if module.params.get('vpc_endpoint_type') != 'Gateway' and module.params.get('route_table_ids'):
        module.fail_json(msg='Route table IDs are only supported for Gateway type VPC Endpoint.')
    if module.check_mode:
        changed = True
        result = 'Would have created VPC Endpoint if not in check mode'
        module.exit_json(changed=changed, result=result)
    if module.params.get('route_table_ids'):
        params['RouteTableIds'] = module.params.get('route_table_ids')
    if module.params.get('vpc_endpoint_subnets'):
        params['SubnetIds'] = module.params.get('vpc_endpoint_subnets')
    if module.params.get('vpc_endpoint_security_groups'):
        params['SecurityGroupIds'] = module.params.get('vpc_endpoint_security_groups')
    if module.params.get('client_token'):
        token_provided = True
        request_time = datetime.datetime.utcnow()
        params['ClientToken'] = module.params.get('client_token')
    policy = None
    if module.params.get('policy'):
        try:
            policy = json.loads(module.params.get('policy'))
        except ValueError as e:
            module.fail_json(msg=str(e), exception=traceback.format_exc(), **camel_dict_to_snake_dict(e.response))
    if policy:
        params['PolicyDocument'] = json.dumps(policy)
    if module.params.get('tags'):
        params['TagSpecifications'] = boto3_tag_specifications(module.params.get('tags'), ['vpc-endpoint'])
    try:
        changed = True
        result = client.create_vpc_endpoint(aws_retry=True, **params)['VpcEndpoint']
        if token_provided and request_time > result['creation_timestamp'].replace(tzinfo=None):
            changed = False
        elif module.params.get('wait') and (not module.check_mode):
            try:
                waiter = get_waiter(client, 'vpc_endpoint_exists')
                waiter.wait(VpcEndpointIds=[result['VpcEndpointId']], WaiterConfig=dict(Delay=15, MaxAttempts=module.params.get('wait_timeout') // 15))
            except botocore.exceptions.WaiterError as e:
                module.fail_json_aws(msg='Error waiting for vpc endpoint to become available - please check the AWS console')
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                module.fail_json_aws(e, msg='Failure while waiting for status')
    except is_boto3_error_code('IdempotentParameterMismatch'):
        module.fail_json(msg='IdempotentParameterMismatch - updates of endpoints are not allowed by the API')
    except is_boto3_error_code('RouteAlreadyExists'):
        module.fail_json(msg='RouteAlreadyExists for one of the route tables - update is not allowed by the API')
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to create VPC.')
    normalized_result = get_endpoints(client, module, endpoint_id=result['VpcEndpointId'])['VpcEndpoints'][0]
    return (changed, normalized_result)
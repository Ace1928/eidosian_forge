import time
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def attach_vgw(client, module, vpn_gateway_id):
    params = dict()
    params['VpcId'] = module.params.get('vpc_id')
    try:
        response = VGWRetry.jittered_backoff(retries=5, catch_extra_error_codes=['InvalidParameterValue'])(client.attach_vpn_gateway)(VpnGatewayId=vpn_gateway_id, VpcId=params['VpcId'])
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to attach VPC')
    status_achieved, vgw = wait_for_status(client, module, [vpn_gateway_id], 'attached')
    if not status_achieved:
        module.fail_json(msg='Error waiting for vpc to attach to vgw - please check the AWS console')
    result = response
    return result
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_eigw(module, connection, vpc_id):
    """
    Create EIGW.

    module       : AnsibleAWSModule object
    connection   : boto3 client connection object
    vpc_id       : ID of the VPC we are operating on
    """
    gateway_id = None
    changed = False
    try:
        response = connection.create_egress_only_internet_gateway(aws_retry=True, DryRun=module.check_mode, VpcId=vpc_id)
    except is_boto3_error_code('DryRunOperation'):
        changed = True
    except is_boto3_error_code('InvalidVpcID.NotFound') as e:
        module.fail_json_aws(e, msg=f"invalid vpc ID '{vpc_id}' provided")
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg=f'Could not create Egress-Only Internet Gateway for vpc ID {vpc_id}')
    if not module.check_mode:
        gateway = response.get('EgressOnlyInternetGateway', {})
        state = gateway.get('Attachments', [{}])[0].get('State')
        gateway_id = gateway.get('EgressOnlyInternetGatewayId')
        if gateway_id and state in ('attached', 'attaching'):
            changed = True
        else:
            module.fail_json(msg=f'Unable to create and attach Egress Only Internet Gateway to VPCId: {vpc_id}. Bad or no state in response', **camel_dict_to_snake_dict(response))
    return (changed, gateway_id)
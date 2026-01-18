from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def delete_eigw(module, connection, eigw_id):
    """
    Delete EIGW.

    module     : AnsibleAWSModule object
    connection : boto3 client connection object
    eigw_id    : ID of the EIGW to delete
    """
    changed = False
    try:
        response = connection.delete_egress_only_internet_gateway(aws_retry=True, DryRun=module.check_mode, EgressOnlyInternetGatewayId=eigw_id)
    except is_boto3_error_code('DryRunOperation'):
        changed = True
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg=f'Could not delete Egress-Only Internet Gateway {eigw_id} from VPC {module.vpc_id}')
    if not module.check_mode:
        changed = response.get('ReturnCode', False)
    return changed
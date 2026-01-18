from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def allocate_address_from_pool(ec2, module, domain, check_mode, public_ipv4_pool, tags):
    """Overrides botocore's allocate_address function to support BYOIP"""
    if check_mode:
        return None
    params = {}
    if domain is not None:
        params['Domain'] = domain
    if public_ipv4_pool is not None:
        params['PublicIpv4Pool'] = public_ipv4_pool
    if tags:
        params['TagSpecifications'] = boto3_tag_specifications(tags, types='elastic-ip')
    try:
        result = ec2.allocate_address(aws_retry=True, **params)
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, msg="Couldn't allocate Elastic IP address")
    return result
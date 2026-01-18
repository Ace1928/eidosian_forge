from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import normalize_ec2_vpc_dhcp_config
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
def fetch_dhcp_options_for_vpc(client, module, vpc_id):
    try:
        vpcs = client.describe_vpcs(aws_retry=True, VpcIds=[vpc_id])['Vpcs']
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, msg=f'Unable to describe vpc {vpc_id}')
    if len(vpcs) != 1:
        return None
    try:
        dhcp_options = client.describe_dhcp_options(aws_retry=True, DhcpOptionsIds=[vpcs[0]['DhcpOptionsId']])
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, msg=f'Unable to describe dhcp option {vpcs[0]['DhcpOptionsId']}')
    if len(dhcp_options['DhcpOptions']) != 1:
        return None
    return (dhcp_options['DhcpOptions'][0]['DhcpConfigurations'], dhcp_options['DhcpOptions'][0]['DhcpOptionsId'])
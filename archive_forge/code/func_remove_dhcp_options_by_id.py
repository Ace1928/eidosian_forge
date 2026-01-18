from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import normalize_ec2_vpc_dhcp_config
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
def remove_dhcp_options_by_id(client, module, dhcp_options_id):
    changed = False
    try:
        associations = client.describe_vpcs(aws_retry=True, Filters=[{'Name': 'dhcp-options-id', 'Values': [dhcp_options_id]}])
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, msg=f'Unable to describe VPC associations for dhcp option id {dhcp_options_id}')
    if len(associations['Vpcs']) > 0:
        return changed
    changed = True
    if not module.check_mode:
        try:
            client.delete_dhcp_options(aws_retry=True, DhcpOptionsId=dhcp_options_id)
        except is_boto3_error_code('InvalidDhcpOptionsID.NotFound'):
            return False
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            module.fail_json_aws(e, msg=f'Unable to delete dhcp option {dhcp_options_id}')
    return changed
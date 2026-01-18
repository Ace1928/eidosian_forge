from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import normalize_ec2_vpc_dhcp_config
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def get_dhcp_options_info(dhcp_option):
    dhcp_option_info = {'DhcpOptionsId': dhcp_option['DhcpOptionsId'], 'DhcpConfigurations': dhcp_option['DhcpConfigurations'], 'Tags': boto3_tag_list_to_ansible_dict(dhcp_option.get('Tags', [{'Value': '', 'Key': 'Name'}]))}
    return dhcp_option_info
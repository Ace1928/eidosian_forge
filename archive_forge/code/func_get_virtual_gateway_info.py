from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_virtual_gateway_info(virtual_gateway):
    tags = virtual_gateway.get('Tags', [])
    resource_tags = boto3_tag_list_to_ansible_dict(tags)
    virtual_gateway_info = dict(VpnGatewayId=virtual_gateway['VpnGatewayId'], State=virtual_gateway['State'], Type=virtual_gateway['Type'], VpcAttachments=virtual_gateway['VpcAttachments'], Tags=tags, ResourceTags=resource_tags)
    return virtual_gateway_info
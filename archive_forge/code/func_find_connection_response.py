from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def find_connection_response(connections=None):
    """Determine if there is a viable unique match in the connections described. Returns the unique VPN connection if one is found,
    returns None if the connection does not exist, raise an error if multiple matches are found."""
    if not connections or 'VpnConnections' not in connections:
        return None
    elif connections and len(connections['VpnConnections']) > 1:
        viable = []
        for each in connections['VpnConnections']:
            if each['State'] not in ('deleted', 'deleting'):
                viable.append(each)
        if len(viable) == 1:
            return viable[0]
        elif len(viable) == 0:
            return None
        else:
            raise VPNConnectionException(msg='More than one matching VPN connection was found. To modify or delete a VPN please specify vpn_connection_id or add filters.')
    elif connections and len(connections['VpnConnections']) == 1:
        if connections['VpnConnections'][0]['State'] not in ('deleted', 'deleting'):
            return connections['VpnConnections'][0]
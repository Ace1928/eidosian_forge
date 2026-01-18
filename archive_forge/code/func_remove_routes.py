from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def remove_routes(connection, vpn_connection_id, routes_to_remove):
    for route in routes_to_remove:
        try:
            connection.delete_vpn_connection_route(aws_retry=True, VpnConnectionId=vpn_connection_id, DestinationCidrBlock=route)
        except (BotoCoreError, ClientError) as e:
            raise VPNConnectionException(msg=f'Failed to remove route {route} from the VPN connection {vpn_connection_id}.', exception=e)
import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def check_dxgw_association(client, module, gateway_id, virtual_gateway_id=None):
    try:
        if virtual_gateway_id is None:
            resp = client.describe_direct_connect_gateway_associations(directConnectGatewayId=gateway_id)
        else:
            resp = client.describe_direct_connect_gateway_associations(directConnectGatewayId=gateway_id, virtualGatewayId=virtual_gateway_id)
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, msg='Failed to check gateway association')
    return resp
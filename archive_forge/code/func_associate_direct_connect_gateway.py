import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def associate_direct_connect_gateway(client, module, gateway_id):
    params = dict()
    params['virtual_gateway_id'] = module.params.get('virtual_gateway_id')
    try:
        response = client.create_direct_connect_gateway_association(directConnectGatewayId=gateway_id, virtualGatewayId=params['virtual_gateway_id'])
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, 'Failed to associate gateway')
    status_achieved, dxgw = wait_for_status(client, module, gateway_id, params['virtual_gateway_id'], 'associating')
    if not status_achieved:
        module.fail_json(msg='Error waiting for dxgw to attach to vpg - please check the AWS console')
    result = response
    return result
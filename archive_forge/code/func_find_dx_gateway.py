import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def find_dx_gateway(client, module, gateway_id=None):
    params = dict()
    gateways = list()
    if gateway_id is not None:
        params['directConnectGatewayId'] = gateway_id
    while True:
        try:
            resp = client.describe_direct_connect_gateways(**params)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            module.fail_json_aws(e, msg='Failed to describe gateways')
        gateways.extend(resp['directConnectGateways'])
        if 'nextToken' in resp:
            params['nextToken'] = resp['nextToken']
        else:
            break
    if gateways != []:
        count = 0
        for gateway in gateways:
            if module.params.get('name') == gateway['directConnectGatewayName']:
                count += 1
                return gateway
    return None
import traceback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import DirectConnectError
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def find_connection_id(client, connection_id=None, connection_name=None):
    params = {}
    if connection_id:
        params['connectionId'] = connection_id
    try:
        response = describe_connections(client, params)
    except (BotoCoreError, ClientError) as e:
        if connection_id:
            msg = f'Failed to describe DirectConnect ID {connection_id}'
        else:
            msg = 'Failed to describe DirectConnect connections'
        raise DirectConnectError(msg=msg, last_traceback=traceback.format_exc(), exception=e)
    match = []
    if len(response.get('connections', [])) == 1 and connection_id:
        if response['connections'][0]['connectionState'] != 'deleted':
            match.append(response['connections'][0]['connectionId'])
    for conn in response.get('connections', []):
        if connection_name == conn['connectionName'] and conn['connectionState'] != 'deleted':
            match.append(conn['connectionId'])
    if len(match) == 1:
        return match[0]
    else:
        raise DirectConnectError(msg='Could not find a valid DirectConnect connection')
import traceback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import DirectConnectError
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_connection_state(client, connection_id):
    try:
        response = describe_connections(client, dict(connectionId=connection_id))
        return response['connections'][0]['connectionState']
    except (BotoCoreError, ClientError, IndexError) as e:
        raise DirectConnectError(msg=f'Failed to describe DirectConnect connection {connection_id} state', last_traceback=traceback.format_exc(), exception=e)
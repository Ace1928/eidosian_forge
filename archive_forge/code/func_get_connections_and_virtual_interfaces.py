import time
import traceback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import DirectConnectError
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import delete_connection
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import delete_virtual_interface
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import disassociate_connection_and_lag
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_connections_and_virtual_interfaces(client, lag_id):
    virtual_interfaces = describe_virtual_interfaces(client, lag_id)
    connections = lag_status(client, lag_id=lag_id).get('connections', [])
    return (virtual_interfaces, connections)
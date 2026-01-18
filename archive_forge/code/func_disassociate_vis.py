import time
import traceback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import DirectConnectError
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import delete_connection
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import delete_virtual_interface
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import disassociate_connection_and_lag
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def disassociate_vis(client, lag_id, virtual_interfaces):
    for vi in virtual_interfaces:
        delete_virtual_interface(client, vi['virtualInterfaceId'])
        try:
            response = client.delete_virtual_interface(virtualInterfaceId=vi['virtualInterfaceId'])
        except botocore.exceptions.ClientError as e:
            raise DirectConnectError(msg=f'Could not delete virtual interface {vi} to delete link aggregation group {lag_id}.', last_traceback=traceback.format_exc(), exception=e)
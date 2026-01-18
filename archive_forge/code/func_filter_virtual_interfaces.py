import traceback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import DirectConnectError
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import delete_virtual_interface
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def filter_virtual_interfaces(virtual_interfaces, name, connection_id):
    """
    Filters the available virtual interfaces to try to find a unique match
    """
    if name:
        matching_by_name = find_virtual_interface_by_name(virtual_interfaces, name)
        if len(matching_by_name) == 1:
            return matching_by_name
    else:
        matching_by_name = virtual_interfaces
    if connection_id and len(matching_by_name) > 1:
        matching_by_connection_id = find_virtual_interface_by_connection_id(matching_by_name, connection_id)
        if len(matching_by_connection_id) == 1:
            return matching_by_connection_id
    else:
        matching_by_connection_id = matching_by_name
    return matching_by_connection_id
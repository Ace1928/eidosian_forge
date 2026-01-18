import traceback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import DirectConnectError
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import delete_virtual_interface
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def find_unique_vi(client, connection_id, virtual_interface_id, name):
    """
    Determines if the virtual interface exists. Returns the virtual interface ID if an exact match is found.
    If multiple matches are found False is returned. If no matches are found None is returned.
    """
    vi_params = {}
    if virtual_interface_id:
        vi_params = {'virtualInterfaceId': virtual_interface_id}
    virtual_interfaces = try_except_ClientError(failure_msg='Failed to describe virtual interface')(client.describe_virtual_interfaces)(**vi_params).get('virtualInterfaces')
    virtual_interfaces = [vi for vi in virtual_interfaces if vi['virtualInterfaceState'] not in ('deleting', 'deleted')]
    matching_virtual_interfaces = filter_virtual_interfaces(virtual_interfaces, name, connection_id)
    return exact_match(matching_virtual_interfaces)
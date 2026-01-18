import traceback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import DirectConnectError
from ansible_collections.amazon.aws.plugins.module_utils.direct_connect import delete_virtual_interface
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_vi(client, public, associated_id, creation_params):
    """
    :param public: a boolean
    :param associated_id: a link aggregation group ID or connection ID to associate
                          with the virtual interface.
    :param creation_params: a dict of parameters to use in the AWS SDK call
    :return The ID of the created virtual interface
    """
    err_msg = 'Failed to create virtual interface'
    if public:
        vi = try_except_ClientError(failure_msg=err_msg)(client.create_public_virtual_interface)(connectionId=associated_id, newPublicVirtualInterface=creation_params)
    else:
        vi = try_except_ClientError(failure_msg=err_msg)(client.create_private_virtual_interface)(connectionId=associated_id, newPrivateVirtualInterface=creation_params)
    return vi['virtualInterfaceId']
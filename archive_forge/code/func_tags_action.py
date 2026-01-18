import time
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def tags_action(client, stream_name, tags, action='create', check_mode=False):
    """Create or delete multiple tags from a Kinesis Stream.
    Args:
        client (botocore.client.EC2): Boto3 client.
        resource_id (str): The Amazon resource id.
        tags (list): List of dictionaries.
            examples.. [{Name: "", Values: [""]}]

    Kwargs:
        action (str): The action to perform.
            valid actions == create and delete
            default=create
        check_mode (bool): This will pass DryRun as one of the parameters to the aws api.
            default=False

    Basic Usage:
        >>> client = boto3.client('ec2')
        >>> resource_id = 'pcx-123345678'
        >>> tags = {'env': 'development'}
        >>> update_tags(client, resource_id, tags)
        [True, '']

    Returns:
        List (bool, str)
    """
    success = False
    err_msg = ''
    params = {'StreamName': stream_name}
    try:
        if not check_mode:
            if action == 'create':
                params['Tags'] = tags
                client.add_tags_to_stream(**params)
                success = True
            elif action == 'delete':
                params['TagKeys'] = tags
                client.remove_tags_from_stream(**params)
                success = True
            else:
                err_msg = f'Invalid action {action}'
        elif action == 'create':
            success = True
        elif action == 'delete':
            success = True
        else:
            err_msg = f'Invalid action {action}'
    except botocore.exceptions.ClientError as e:
        err_msg = to_native(e)
    return (success, err_msg)
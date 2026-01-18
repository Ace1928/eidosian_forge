from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import get_boto3_client_method_parameters
from ansible_collections.amazon.aws.plugins.module_utils.exceptions import AnsibleAWSError
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def do_create_project(client, params, formatted_params):
    if params['source'] is None or params['artifacts'] is None:
        raise CodeBuildAnsibleAWSError(message='The source and artifacts parameters must be provided when creating a new project.  No existing project was found.')
    if params['tags'] is not None:
        formatted_params['tags'] = ansible_dict_to_boto3_tag_list(params['tags'], tag_name_key_name='key', tag_value_key_name='value')
    permitted_create_params = get_boto3_client_method_parameters(client, 'create_project')
    formatted_create_params = dict(((k, v) for k, v in formatted_params.items() if k in permitted_create_params))
    try:
        resp = client.create_project(**formatted_create_params)
        changed = True
        return (resp, changed)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        raise CodeBuildAnsibleAWSError(message='Unable to create CodeBuild project', exception=e)
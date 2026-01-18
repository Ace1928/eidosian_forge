from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import get_boto3_client_method_parameters
from ansible_collections.amazon.aws.plugins.module_utils.exceptions import AnsibleAWSError
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def do_update_project(client, params, formatted_params, found_project):
    permitted_update_params = get_boto3_client_method_parameters(client, 'update_project')
    formatted_update_params = dict(((k, v) for k, v in formatted_params.items() if k in permitted_update_params))
    found_tags = found_project.pop('tags', [])
    if params['tags'] is not None:
        formatted_update_params['tags'] = format_tags(merge_tags(found_tags, params['tags'], params['purge_tags']))
    resp = update_project(client=client, params=formatted_update_params)
    updated_project = resp['project']
    found_project.pop('lastModified')
    updated_project.pop('lastModified')
    updated_tags = updated_project.pop('tags', [])
    found_project['ResourceTags'] = boto3_tag_list_to_ansible_dict(found_tags)
    updated_project['ResourceTags'] = boto3_tag_list_to_ansible_dict(updated_tags)
    changed = updated_project != found_project
    updated_project['tags'] = updated_tags
    return (resp, changed)
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import get_boto3_client_method_parameters
from ansible_collections.amazon.aws.plugins.module_utils.exceptions import AnsibleAWSError
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def format_project_result(project_result):
    formated_result = camel_dict_to_snake_dict(project_result)
    project = project_result.get('project', {})
    if project:
        tags = project.get('tags', [])
        formated_result['project']['resource_tags'] = boto3_tag_list_to_ansible_dict(tags)
    formated_result['ORIGINAL'] = project_result
    return formated_result
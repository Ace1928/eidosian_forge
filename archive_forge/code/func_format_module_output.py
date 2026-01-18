from uuid import uuid4
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.arn import validate_aws_arn
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def format_module_output(module):
    output = {}
    template, template_versions = existing_templates(module)
    template = camel_dict_to_snake_dict(template)
    template_versions = [camel_dict_to_snake_dict(v) for v in template_versions]
    for v in template_versions:
        for ts in v['launch_template_data'].get('tag_specifications') or []:
            ts['tags'] = boto3_tag_list_to_ansible_dict(ts.pop('tags'))
    output.update(dict(template=template, versions=template_versions))
    output['default_template'] = [v for v in template_versions if v.get('default_version')][0]
    output['latest_template'] = [v for v in template_versions if v.get('version_number') and int(v['version_number']) == int(template['latest_version_number'])][0]
    if 'version_number' in output['default_template']:
        output['default_version'] = output['default_template']['version_number']
    if 'version_number' in output['latest_template']:
        output['latest_version'] = output['latest_template']['version_number']
    return output
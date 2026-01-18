import json
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def key_matches_filter(key, filtr):
    if filtr[0] == 'key-id':
        return filtr[1] == key['key_id']
    if filtr[0] == 'tag-key':
        return filtr[1] in key['tags']
    if filtr[0] == 'tag-value':
        return filtr[1] in key['tags'].values()
    if filtr[0] == 'alias':
        return filtr[1] in key['aliases']
    if filtr[0].startswith('tag:'):
        tag_key = filtr[0][4:]
        if tag_key not in key['tags']:
            return False
        return key['tags'].get(tag_key) == filtr[1]
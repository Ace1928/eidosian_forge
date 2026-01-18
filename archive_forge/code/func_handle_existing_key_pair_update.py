import os
import uuid
from ansible.module_utils._text import to_bytes
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
def handle_existing_key_pair_update(module, ec2_client, name, key):
    key_material = module.params.get('key_material')
    force = module.params.get('force')
    key_type = module.params.get('key_type')
    tags = module.params.get('tags')
    purge_tags = module.params.get('purge_tags')
    tag_spec = boto3_tag_specifications(tags, ['key-pair'])
    check_mode = module.check_mode
    file_name = module.params.get('file_name')
    if key_material and force:
        result = update_key_pair_by_key_material(check_mode, ec2_client, name, key, key_material, tag_spec)
    elif key_type and key_type != key['KeyType']:
        result = update_key_pair_by_key_type(check_mode, ec2_client, name, key_type, tag_spec, file_name)
    else:
        changed = False
        changed |= ensure_ec2_tags(ec2_client, module, key['KeyPairId'], tags=tags, purge_tags=purge_tags)
        key = find_key_pair(ec2_client, name)
        key_data = extract_key_data(key, file_name=file_name)
        result = {'changed': changed, 'key': key_data, 'msg': 'key pair already exists'}
    return result
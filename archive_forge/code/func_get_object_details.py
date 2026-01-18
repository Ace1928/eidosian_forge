from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.s3 import s3_extra_params
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def get_object_details(connection, module, bucket_name, object_name, requested_facts):
    all_facts = {}
    requested_facts = {fact: value for fact, value in requested_facts.items() if value is True}
    all_facts['object_data'] = get_object(connection, bucket_name, object_name)['object_data']
    all_facts['object_name'] = object_name
    for key in requested_facts:
        if key == 'object_acl':
            all_facts[key] = {}
            all_facts[key] = describe_s3_object_acl(connection, bucket_name, object_name)
        elif key == 'object_attributes':
            all_facts[key] = {}
            all_facts[key] = describe_s3_object_attributes(connection, module, bucket_name, object_name)
        elif key == 'object_legal_hold':
            all_facts[key] = {}
            all_facts[key] = describe_s3_object_legal_hold(connection, bucket_name, object_name)
        elif key == 'object_lock_configuration':
            all_facts[key] = {}
            all_facts[key] = describe_s3_object_lock_configuration(connection, bucket_name)
        elif key == 'object_retention':
            all_facts[key] = {}
            all_facts[key] = describe_s3_object_retention(connection, bucket_name, object_name)
        elif key == 'object_tagging':
            all_facts[key] = {}
            all_facts[key] = describe_s3_object_tagging(connection, bucket_name, object_name)
    return all_facts
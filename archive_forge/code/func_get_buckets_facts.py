from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def get_buckets_facts(connection, buckets, requested_facts, transform_location):
    """
    Retrieve additional information about S3 buckets
    """
    full_bucket_list = []
    for bucket in buckets:
        bucket.update(get_bucket_details(connection, bucket['name'], requested_facts, transform_location))
        full_bucket_list.append(bucket)
    return full_bucket_list
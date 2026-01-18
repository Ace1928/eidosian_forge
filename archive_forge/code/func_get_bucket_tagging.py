from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
@AWSRetry.jittered_backoff(max_delay=120, catch_extra_error_codes=['NoSuchBucket', 'OperationAborted'])
def get_bucket_tagging(name, connection):
    """
    Get bucket tags and transform them using `boto3_tag_list_to_ansible_dict` function
    """
    data = connection.get_bucket_tagging(Bucket=name)
    try:
        bucket_tags = boto3_tag_list_to_ansible_dict(data['TagSet'])
        return bucket_tags
    except KeyError:
        data.pop('ResponseMetadata', None)
        return data
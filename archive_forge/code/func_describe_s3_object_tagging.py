from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.s3 import s3_extra_params
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def describe_s3_object_tagging(connection, bucket_name, object_name):
    params = {}
    params['Bucket'] = bucket_name
    params['Key'] = object_name
    object_tagging_info = {}
    try:
        object_tagging_info = connection.get_object_tagging(**params)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        pass
    if len(object_tagging_info) != 0:
        del object_tagging_info['ResponseMetadata']
        object_tagging_info = boto3_tag_list_to_ansible_dict(object_tagging_info['TagSet'])
    return object_tagging_info
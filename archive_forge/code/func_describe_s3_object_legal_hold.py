from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.s3 import s3_extra_params
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def describe_s3_object_legal_hold(connection, bucket_name, object_name):
    params = {}
    params['Bucket'] = bucket_name
    params['Key'] = object_name
    object_legal_hold_info = {}
    try:
        object_legal_hold_info = connection.get_object_legal_hold(**params)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        pass
    if len(object_legal_hold_info) != 0:
        del object_legal_hold_info['ResponseMetadata']
        object_legal_hold_info = camel_dict_to_snake_dict(object_legal_hold_info)
    return object_legal_hold_info
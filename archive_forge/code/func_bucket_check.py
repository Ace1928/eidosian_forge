from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.s3 import s3_extra_params
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def bucket_check(connection, module, bucket_name):
    try:
        connection.head_bucket(Bucket=bucket_name)
    except is_boto3_error_code(['404', '403']) as e:
        module.fail_json_aws(e, msg=f'The bucket {bucket_name} does not exist or is missing access permissions.')
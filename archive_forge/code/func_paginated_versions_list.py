import json
import time
from ansible.module_utils.basic import to_text
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.s3 import s3_extra_params
from ansible_collections.amazon.aws.plugins.module_utils.s3 import validate_bucket_name
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def paginated_versions_list(s3_client, **pagination_params):
    try:
        pg = s3_client.get_paginator('list_object_versions')
        for page in pg.paginate(**pagination_params):
            yield [(data['Key'], data['VersionId']) for data in page.get('Versions', []) + page.get('DeleteMarkers', [])]
    except is_boto3_error_code('NoSuchBucket'):
        yield []
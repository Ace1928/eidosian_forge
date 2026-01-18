from collections import defaultdict
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
@AWSRetry.exponential_backoff(catch_extra_error_codes=['ThrottlingException'])
def list_file_systems(self, **kwargs):
    """
        Returns generator of file systems including all attributes of FS
        """
    paginator = self.connection.get_paginator('describe_file_systems')
    return paginator.paginate(**kwargs).build_full_result()['FileSystems']
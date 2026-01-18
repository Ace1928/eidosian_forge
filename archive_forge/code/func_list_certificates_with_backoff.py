from ansible.module_utils._text import to_bytes
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from .botocore import is_boto3_error_code
from .retries import AWSRetry
from .tagging import ansible_dict_to_boto3_tag_list
from .tagging import boto3_tag_list_to_ansible_dict
@acm_catch_boto_exception
@AWSRetry.jittered_backoff(delay=5, catch_extra_error_codes=['RequestInProgressException'])
def list_certificates_with_backoff(self, statuses=None):
    paginator = self.client.get_paginator('list_certificates')
    kwargs = {'Includes': {'keyTypes': ['RSA_1024', 'RSA_2048', 'RSA_3072', 'RSA_4096', 'EC_prime256v1', 'EC_secp384r1', 'EC_secp521r1']}}
    if statuses:
        kwargs['CertificateStatuses'] = statuses
    return paginator.paginate(**kwargs).build_full_result()['CertificateSummaryList']
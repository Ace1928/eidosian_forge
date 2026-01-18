from ansible.module_utils._text import to_bytes
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from .botocore import is_boto3_error_code
from .retries import AWSRetry
from .tagging import ansible_dict_to_boto3_tag_list
from .tagging import boto3_tag_list_to_ansible_dict
@AWSRetry.jittered_backoff(delay=5, catch_extra_error_codes=['RequestInProgressException', 'ResourceNotFoundException'])
def tag_certificate_with_backoff(self, arn, tags):
    aws_tags = ansible_dict_to_boto3_tag_list(tags)
    self.client.add_tags_to_certificate(CertificateArn=arn, Tags=aws_tags)
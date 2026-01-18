from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.iam import AnsibleIAMError
from ansible_collections.amazon.aws.plugins.module_utils.iam import IAMErrorHandler
from ansible_collections.amazon.aws.plugins.module_utils.iam import convert_managed_policy_names_to_arns
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_iam_user
from ansible_collections.amazon.aws.plugins.module_utils.iam import normalize_iam_user
from ansible_collections.amazon.aws.plugins.module_utils.iam import validate_iam_identifiers
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
@IAMErrorHandler.list_error_handler('list signing certificates')
def delete_signing_certificates(connection, check_mode, user_name):
    certificates = connection.list_signing_certificates(aws_retry=True, UserName=user_name)['Certificates']
    if not certificates:
        return False
    for certificate in certificates:
        delete_signing_certificate(connection, check_mode, user_name, certificate['CertificateId'])
    return True
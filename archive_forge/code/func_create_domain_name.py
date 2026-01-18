import copy
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
@AWSRetry.jittered_backoff(**retry_params)
def create_domain_name(module, client, domain_name, certificate_arn, endpoint_type, security_policy):
    endpoint_configuration = {'types': [endpoint_type]}
    if endpoint_type == 'EDGE':
        return client.create_domain_name(domainName=domain_name, certificateArn=certificate_arn, endpointConfiguration=endpoint_configuration, securityPolicy=security_policy)
    else:
        return client.create_domain_name(domainName=domain_name, regionalCertificateArn=certificate_arn, endpointConfiguration=endpoint_configuration, securityPolicy=security_policy)
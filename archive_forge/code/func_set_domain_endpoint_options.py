import datetime
import json
from copy import deepcopy
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
from ansible_collections.community.aws.plugins.module_utils.opensearch import compare_domain_versions
from ansible_collections.community.aws.plugins.module_utils.opensearch import ensure_tags
from ansible_collections.community.aws.plugins.module_utils.opensearch import get_domain_config
from ansible_collections.community.aws.plugins.module_utils.opensearch import get_domain_status
from ansible_collections.community.aws.plugins.module_utils.opensearch import get_target_increment_version
from ansible_collections.community.aws.plugins.module_utils.opensearch import normalize_opensearch
from ansible_collections.community.aws.plugins.module_utils.opensearch import parse_version
from ansible_collections.community.aws.plugins.module_utils.opensearch import wait_for_domain_status
def set_domain_endpoint_options(module, current_domain_config, desired_domain_config, change_set):
    changed = False
    domain_endpoint_config = desired_domain_config['DomainEndpointOptions']
    domain_endpoint_opts = module.params.get('domain_endpoint_options')
    if domain_endpoint_opts is None:
        return changed
    if domain_endpoint_opts.get('enforce_https') is not None:
        domain_endpoint_config['EnforceHTTPS'] = domain_endpoint_opts.get('enforce_https')
    if domain_endpoint_opts.get('tls_security_policy') is not None:
        domain_endpoint_config['TLSSecurityPolicy'] = domain_endpoint_opts.get('tls_security_policy')
    if domain_endpoint_opts.get('custom_endpoint_enabled') is not None:
        domain_endpoint_config['CustomEndpointEnabled'] = domain_endpoint_opts.get('custom_endpoint_enabled')
    if domain_endpoint_config['CustomEndpointEnabled']:
        if domain_endpoint_opts.get('custom_endpoint') is not None:
            domain_endpoint_config['CustomEndpoint'] = domain_endpoint_opts.get('custom_endpoint')
        if domain_endpoint_opts.get('custom_endpoint_certificate_arn') is not None:
            domain_endpoint_config['CustomEndpointCertificateArn'] = domain_endpoint_opts.get('custom_endpoint_certificate_arn')
    if current_domain_config is not None and current_domain_config['DomainEndpointOptions'] != domain_endpoint_config:
        change_set.append(f'DomainEndpointOptions changed from {current_domain_config['DomainEndpointOptions']} to {domain_endpoint_config}')
        changed = True
    return changed
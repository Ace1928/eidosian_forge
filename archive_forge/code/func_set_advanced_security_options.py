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
def set_advanced_security_options(module, current_domain_config, desired_domain_config, change_set):
    changed = False
    advanced_security_config = desired_domain_config['AdvancedSecurityOptions']
    advanced_security_opts = module.params.get('advanced_security_options')
    if advanced_security_opts is None:
        return changed
    if advanced_security_opts.get('enabled') is not None:
        advanced_security_config['Enabled'] = advanced_security_opts.get('enabled')
    if not advanced_security_config['Enabled']:
        desired_domain_config['AdvancedSecurityOptions'] = {'Enabled': False}
    else:
        if advanced_security_opts.get('internal_user_database_enabled') is not None:
            advanced_security_config['InternalUserDatabaseEnabled'] = advanced_security_opts.get('internal_user_database_enabled')
        master_user_opts = advanced_security_opts.get('master_user_options')
        if master_user_opts is not None:
            advanced_security_config.setdefault('MasterUserOptions', {})
            if master_user_opts.get('master_user_arn') is not None:
                advanced_security_config['MasterUserOptions']['MasterUserARN'] = master_user_opts.get('master_user_arn')
            if master_user_opts.get('master_user_name') is not None:
                advanced_security_config['MasterUserOptions']['MasterUserName'] = master_user_opts.get('master_user_name')
            if master_user_opts.get('master_user_password') is not None:
                advanced_security_config['MasterUserOptions']['MasterUserPassword'] = master_user_opts.get('master_user_password')
        saml_opts = advanced_security_opts.get('saml_options')
        if saml_opts is not None:
            if saml_opts.get('enabled') is not None:
                advanced_security_config['SamlOptions']['Enabled'] = saml_opts.get('enabled')
            idp_opts = saml_opts.get('idp')
            if idp_opts is not None:
                if idp_opts.get('metadata_content') is not None:
                    advanced_security_config['SamlOptions']['Idp']['MetadataContent'] = idp_opts.get('metadata_content')
                if idp_opts.get('entity_id') is not None:
                    advanced_security_config['SamlOptions']['Idp']['EntityId'] = idp_opts.get('entity_id')
            if saml_opts.get('master_user_name') is not None:
                advanced_security_config['SamlOptions']['MasterUserName'] = saml_opts.get('master_user_name')
            if saml_opts.get('master_backend_role') is not None:
                advanced_security_config['SamlOptions']['MasterBackendRole'] = saml_opts.get('master_backend_role')
            if saml_opts.get('subject_key') is not None:
                advanced_security_config['SamlOptions']['SubjectKey'] = saml_opts.get('subject_key')
            if saml_opts.get('roles_key') is not None:
                advanced_security_config['SamlOptions']['RolesKey'] = saml_opts.get('roles_key')
            if saml_opts.get('session_timeout_minutes') is not None:
                advanced_security_config['SamlOptions']['SessionTimeoutMinutes'] = saml_opts.get('session_timeout_minutes')
    if current_domain_config is not None and current_domain_config['AdvancedSecurityOptions'] != advanced_security_config:
        change_set.append(f'AdvancedSecurityOptions changed from {current_domain_config['AdvancedSecurityOptions']} to {advanced_security_config}')
        changed = True
    return changed
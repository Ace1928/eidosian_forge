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
def set_access_policy(module, current_domain_config, desired_domain_config, change_set):
    access_policy_config = None
    changed = False
    access_policy_opt = module.params.get('access_policies')
    if access_policy_opt is None:
        return changed
    try:
        access_policy_config = json.dumps(access_policy_opt)
    except Exception as e:
        module.fail_json(msg=f'Failed to convert the policy into valid JSON: {str(e)}')
    if current_domain_config is not None:
        current_access_policy = json.loads(current_domain_config['AccessPolicies'])
        if not compare_policies(current_access_policy, access_policy_opt):
            change_set.append(f'AccessPolicy changed from {current_access_policy} to {access_policy_opt}')
            changed = True
            desired_domain_config['AccessPolicies'] = access_policy_config
    else:
        desired_domain_config['AccessPolicies'] = access_policy_config
    return changed
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
def set_node_to_node_encryption_options(module, current_domain_config, desired_domain_config, change_set):
    changed = False
    node_to_node_encryption_config = desired_domain_config['NodeToNodeEncryptionOptions']
    node_to_node_encryption_opts = module.params.get('node_to_node_encryption_options')
    if node_to_node_encryption_opts is None:
        return changed
    if node_to_node_encryption_opts.get('enabled') is not None:
        node_to_node_encryption_config['Enabled'] = node_to_node_encryption_opts.get('enabled')
    if current_domain_config is not None and current_domain_config['NodeToNodeEncryptionOptions'] != node_to_node_encryption_config:
        change_set.append(f'NodeToNodeEncryptionOptions changed from {current_domain_config['NodeToNodeEncryptionOptions']} to {node_to_node_encryption_config}')
        changed = True
    return changed
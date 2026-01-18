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
def set_snapshot_options(module, current_domain_config, desired_domain_config, change_set):
    changed = False
    snapshot_config = desired_domain_config['SnapshotOptions']
    snapshot_opts = module.params.get('snapshot_options')
    if snapshot_opts is None:
        return changed
    if snapshot_opts.get('automated_snapshot_start_hour') is not None:
        snapshot_config['AutomatedSnapshotStartHour'] = snapshot_opts.get('automated_snapshot_start_hour')
    if current_domain_config is not None and current_domain_config['SnapshotOptions'] != snapshot_config:
        change_set.append('SnapshotOptions changed')
        changed = True
    return changed
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
def ensure_domain_absent(client, module):
    domain_name = module.params.get('domain_name')
    changed = False
    domain = get_domain_status(client, module, domain_name)
    if module.check_mode:
        module.exit_json(changed=True, msg='Would have deleted domain if not in check mode')
    try:
        client.delete_domain(DomainName=domain_name)
        changed = True
    except is_boto3_error_code('ResourceNotFoundException'):
        return dict(changed=False)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='trying to delete domain')
    if not domain or not module.params.get('wait'):
        return dict(changed=changed)
    try:
        wait_for_domain_status(client, module, domain_name, 'domain_deleted')
        return dict(changed=changed)
    except is_boto3_error_code('ResourceNotFoundException'):
        return dict(changed=changed)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, 'awaiting domain deletion')
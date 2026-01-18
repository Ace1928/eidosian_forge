import re
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_regional_rules_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_regional_web_acls_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_rules_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_web_acls_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import run_func_with_change_token_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_web_acl_by_name(client, module, name):
    acls = [d['WebACLId'] for d in list_web_acls(client, module) if d['Name'] == name]
    if acls:
        return acls[0]
    else:
        return acls
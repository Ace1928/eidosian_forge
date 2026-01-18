import re
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.waf import MATCH_LOOKUP
from ansible_collections.amazon.aws.plugins.module_utils.waf import get_web_acl_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_regional_rules_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_regional_web_acls_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_rules_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_web_acls_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import run_func_with_change_token_backoff
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def format_for_insertion(condition):
    return dict(Action='INSERT', Predicate=dict(Negated=condition['negated'], Type=MATCH_LOOKUP[condition['type']]['type'], DataId=condition['data_id']))
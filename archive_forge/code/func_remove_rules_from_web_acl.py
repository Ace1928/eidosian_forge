import re
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_regional_rules_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_regional_web_acls_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_rules_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_web_acls_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import run_func_with_change_token_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def remove_rules_from_web_acl(client, module, web_acl_id):
    acl = get_web_acl(client, module, web_acl_id)
    deletions = [format_for_update(rule, 'DELETE') for rule in acl['Rules']]
    try:
        params = {'WebACLId': acl['WebACLId'], 'DefaultAction': acl['DefaultAction'], 'Updates': deletions}
        run_func_with_change_token_backoff(client, module, params, client.update_web_acl)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Could not remove rule')
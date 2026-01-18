import re
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_regional_rules_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_regional_web_acls_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_rules_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_web_acls_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import run_func_with_change_token_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def ensure_web_acl_present(client, module):
    changed = False
    result = None
    name = module.params['name']
    web_acl_id = get_web_acl_by_name(client, module, name)
    if web_acl_id:
        changed, result = find_and_update_web_acl(client, module, web_acl_id)
    else:
        metric_name = module.params['metric_name']
        if not metric_name:
            metric_name = re.sub('[^A-Za-z0-9]', '', module.params['name'])
        default_action = module.params['default_action'].upper()
        try:
            params = {'Name': name, 'MetricName': metric_name, 'DefaultAction': {'Type': default_action}}
            new_web_acl = run_func_with_change_token_backoff(client, module, params, client.create_web_acl)
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg='Could not create Web ACL')
        changed, result = find_and_update_web_acl(client, module, new_web_acl['WebACL']['WebACLId'])
    return (changed, result)
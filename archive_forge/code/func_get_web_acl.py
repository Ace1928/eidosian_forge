from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from .retries import AWSRetry
from .waiters import get_waiter
def get_web_acl(client, module, web_acl_id):
    try:
        web_acl = get_web_acl_with_backoff(client, web_acl_id)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg="Couldn't obtain web acl")
    if web_acl:
        try:
            for rule in web_acl['Rules']:
                rule.update(get_rule(client, module, rule['RuleId']))
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg="Couldn't obtain web acl rule")
    return camel_dict_to_snake_dict(web_acl)
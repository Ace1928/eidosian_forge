import json
import re
from ansible.module_utils._text import to_native
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def remove_policy_permission(module, client):
    """
    Removed a permission statement from the policy.

    :param module:
    :param aws:
    :return:
    """
    changed = False
    api_params = set_api_params(module, ('function_name', 'statement_id'))
    qualifier = get_qualifier(module)
    if qualifier:
        api_params.update(Qualifier=qualifier)
    try:
        if not module.check_mode:
            client.remove_permission(**api_params)
            changed = True
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='removing permission from policy')
    return changed
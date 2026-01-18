import json
import re
from ansible.module_utils._text import to_native
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def add_policy_permission(module, client):
    """
    Adds a permission statement to the policy.

    :param module:
    :param aws:
    :return:
    """
    changed = False
    params = ('function_name', 'statement_id', 'action', 'principal', 'source_arn', 'source_account', 'event_source_token')
    api_params = set_api_params(module, params)
    qualifier = get_qualifier(module)
    if qualifier:
        api_params.update(Qualifier=qualifier)
    if not module.check_mode:
        try:
            client.add_permission(**api_params)
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg='adding permission to policy')
        changed = True
    return changed
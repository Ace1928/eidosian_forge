import json
import re
from ansible.module_utils._text import to_native
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def get_policy_statement(module, client):
    """Checks that policy exists and if so, that statement ID is present or absent.

    :param module:
    :param client:
    :return:
    """
    sid = module.params['statement_id']
    api_params = set_api_params(module, ('function_name',))
    qualifier = get_qualifier(module)
    if qualifier:
        api_params.update(Qualifier=qualifier)
    policy_results = None
    try:
        policy_results = client.get_policy(**api_params)
    except is_boto3_error_code('ResourceNotFoundException'):
        return {}
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='retrieving function policy')
    policy = json.loads(policy_results.get('Policy', '{}'))
    return extract_statement(policy, sid)
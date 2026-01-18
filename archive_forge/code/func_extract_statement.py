import json
import re
from ansible.module_utils._text import to_native
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def extract_statement(policy, sid):
    """return flattened single policy statement from a policy

    If a policy statement is present in the policy extract it and
    return it in a flattened form.  Otherwise return an empty
    dictionary.
    """
    if 'Statement' not in policy:
        return {}
    policy_statement = {}
    for statement in policy['Statement']:
        if statement['Sid'] == sid:
            policy_statement['action'] = statement['Action']
            try:
                policy_statement['principal'] = statement['Principal']['Service']
            except KeyError:
                pass
            try:
                policy_statement['principal'] = statement['Principal']['AWS']
            except KeyError:
                pass
            try:
                policy_statement['source_arn'] = statement['Condition']['ArnLike']['AWS:SourceArn']
            except KeyError:
                pass
            try:
                policy_statement['source_account'] = statement['Condition']['StringEquals']['AWS:SourceAccount']
            except KeyError:
                pass
            try:
                policy_statement['event_source_token'] = statement['Condition']['StringEquals']['lambda:EventSourceToken']
            except KeyError:
                pass
            break
    return policy_statement
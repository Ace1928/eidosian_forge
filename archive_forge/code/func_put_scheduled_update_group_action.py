from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def put_scheduled_update_group_action(current_actions):
    changed = False
    changes = dict()
    params = format_request()
    if len(current_actions) < 1:
        changed = True
    else:
        if 'StartTime' in params:
            params['StartTime'] = timedate_parse(params['StartTime'])
        if 'EndTime' in params:
            params['EndTime'] = timedate_parse(params['EndTime'])
        for k, v in params.items():
            if current_actions[0].get(k) != v:
                changes[k] = v
        if changes:
            changed = True
    if module.check_mode:
        return changed
    try:
        client.put_scheduled_update_group_action(aws_retry=True, **params)
    except botocore.exceptions.ClientError as e:
        module.fail_json_aws(e)
    return changed
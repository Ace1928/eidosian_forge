import time
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.route53 import get_tags
from ansible_collections.amazon.aws.plugins.module_utils.route53 import manage_tags
def delete_public(matching_zones):
    if len(matching_zones) > 1:
        changed = False
        msg = 'There are multiple zones that match. Use hosted_zone_id to specify the correct zone.'
    else:
        if not module.check_mode:
            try:
                client.delete_hosted_zone(Id=matching_zones[0]['Id'])
            except (BotoCoreError, ClientError) as e:
                module.fail_json_aws(e, msg=f'Could not get delete hosted zone {matching_zones[0]['Id']}')
        changed = True
        msg = f'Successfully deleted {matching_zones[0]['Id']}'
    return (changed, msg)
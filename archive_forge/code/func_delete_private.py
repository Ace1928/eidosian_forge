import time
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.route53 import get_tags
from ansible_collections.amazon.aws.plugins.module_utils.route53 import manage_tags
def delete_private(matching_zones, vpcs):
    for z in matching_zones:
        try:
            result = client.get_hosted_zone(Id=z['Id'])
        except (BotoCoreError, ClientError) as e:
            module.fail_json_aws(e, msg=f'Could not get details about hosted zone {z['Id']}')
        zone_details = result['HostedZone']
        vpc_details = result['VPCs']
        if isinstance(vpc_details, dict):
            if vpc_details['VPC']['VPCId'] == vpcs[0]['id'] and vpcs[0]['region'] == vpc_details['VPC']['VPCRegion']:
                if not module.check_mode:
                    try:
                        client.delete_hosted_zone(Id=z['Id'])
                    except (BotoCoreError, ClientError) as e:
                        module.fail_json_aws(e, msg=f'Could not delete hosted zone {z['Id']}')
                return (True, f'Successfully deleted {zone_details['Name']}')
        elif sorted([vpc['id'] for vpc in vpcs]) == sorted([v['VPCId'] for v in vpc_details]) and sorted([vpc['region'] for vpc in vpcs]) == sorted([v['VPCRegion'] for v in vpc_details]):
            if not module.check_mode:
                try:
                    client.delete_hosted_zone(Id=z['Id'])
                except (BotoCoreError, ClientError) as e:
                    module.fail_json_aws(e, msg=f'Could not delete hosted zone {z['Id']}')
            return (True, f'Successfully deleted {zone_details['Name']}')
    return (False, 'The VPCs do not match a private hosted zone.')
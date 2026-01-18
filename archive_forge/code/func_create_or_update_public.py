import time
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.route53 import get_tags
from ansible_collections.amazon.aws.plugins.module_utils.route53 import manage_tags
def create_or_update_public(matching_zones, record):
    zone_details, zone_delegation_set_details = (None, {})
    for matching_zone in matching_zones:
        try:
            zone = client.get_hosted_zone(Id=matching_zone['Id'])
            zone_details = zone['HostedZone']
            zone_delegation_set_details = zone.get('DelegationSet', {})
        except (BotoCoreError, ClientError) as e:
            module.fail_json_aws(e, msg=f'Could not get details about hosted zone {matching_zone['Id']}')
        if 'Comment' in zone_details['Config'] and zone_details['Config']['Comment'] != record['comment']:
            if not module.check_mode:
                try:
                    client.update_hosted_zone_comment(Id=zone_details['Id'], Comment=record['comment'])
                except (BotoCoreError, ClientError) as e:
                    module.fail_json_aws(e, msg=f'Could not update comment for hosted zone {zone_details['Id']}')
            changed = True
        else:
            changed = False
        break
    if zone_details is None:
        if not module.check_mode:
            try:
                params = dict(Name=record['name'], HostedZoneConfig={'Comment': record['comment'] if record['comment'] is not None else '', 'PrivateZone': False}, CallerReference=f'{record['name']}-{time.time()}')
                if record.get('delegation_set_id') is not None:
                    params['DelegationSetId'] = record['delegation_set_id']
                result = client.create_hosted_zone(**params)
                zone_details = result['HostedZone']
                zone_delegation_set_details = result.get('DelegationSet', {})
            except (BotoCoreError, ClientError) as e:
                module.fail_json_aws(e, msg='Could not create hosted zone')
        changed = True
    if module.check_mode:
        if zone_details:
            record['zone_id'] = zone_details['Id'].replace('/hostedzone/', '')
    else:
        record['zone_id'] = zone_details['Id'].replace('/hostedzone/', '')
        record['name'] = zone_details['Name']
        record['delegation_set_id'] = zone_delegation_set_details.get('Id', '').replace('/delegationset/', '')
    return (changed, record)
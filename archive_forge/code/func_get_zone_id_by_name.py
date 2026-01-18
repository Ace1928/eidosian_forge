from operator import itemgetter
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def get_zone_id_by_name(route53, module, zone_name, want_private, want_vpc_id):
    """Finds a zone by name or zone_id"""
    hosted_zones_results = _list_hosted_zones(route53)
    for zone in hosted_zones_results:
        private_zone = module.boolean(zone['Config'].get('PrivateZone', False))
        zone_id = zone['Id'].replace('/hostedzone/', '')
        if private_zone == want_private and zone['Name'] == zone_name:
            if want_vpc_id:
                hosted_zone = route53.get_hosted_zone(aws_retry=True, Id=zone_id)
                if want_vpc_id in [v['VPCId'] for v in hosted_zone['VPCs']]:
                    return zone_id
            else:
                return zone_id
    return None
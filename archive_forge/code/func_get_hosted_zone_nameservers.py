from operator import itemgetter
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def get_hosted_zone_nameservers(route53, zone_id):
    hosted_zone_name = route53.get_hosted_zone(aws_retry=True, Id=zone_id)['HostedZone']['Name']
    resource_records_sets = _list_record_sets(route53, HostedZoneId=zone_id)
    nameservers_records = list(filter(lambda record: record['Name'] == hosted_zone_name and record['Type'] == 'NS', resource_records_sets))[0]['ResourceRecords']
    return [ns_record['Value'] for ns_record in nameservers_records]
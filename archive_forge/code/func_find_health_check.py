import uuid
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.route53 import get_tags
from ansible_collections.amazon.aws.plugins.module_utils.route53 import manage_tags
def find_health_check(ip_addr, fqdn, hc_type, request_interval, port):
    """Searches for health checks that have the exact same set of immutable values"""
    results = _list_health_checks()
    while True:
        for check in results.get('HealthChecks'):
            config = check.get('HealthCheckConfig')
            if config.get('IPAddress', None) == ip_addr and config.get('FullyQualifiedDomainName', None) == fqdn and (config.get('Type') == hc_type) and (config.get('RequestInterval') == request_interval) and (config.get('Port', None) == port):
                return check
        if results.get('IsTruncated', False):
            results = _list_health_checks(Marker=results.get('NextMarker'))
        else:
            return None
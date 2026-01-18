import uuid
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.route53 import get_tags
from ansible_collections.amazon.aws.plugins.module_utils.route53 import manage_tags
def get_existing_checks_with_name():
    results = _list_health_checks()
    health_checks_with_name = {}
    while True:
        for check in results.get('HealthChecks'):
            if 'Name' in describe_health_check(check['Id'])['tags']:
                check_name = describe_health_check(check['Id'])['tags']['Name']
                health_checks_with_name[check_name] = check
        if results.get('IsTruncated', False):
            results = _list_health_checks(Marker=results.get('NextMarker'))
        else:
            return health_checks_with_name
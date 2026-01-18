from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def normalize_results(results):
    """
    We used to be a boto v2 module, make sure that the old return values are
    maintained and the shape of the return values are what people expect
    """
    routes = [normalize_route_table(route) for route in results['RouteTables']]
    del results['RouteTables']
    results = camel_dict_to_snake_dict(results)
    results['route_tables'] = routes
    return results
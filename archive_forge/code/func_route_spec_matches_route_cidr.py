import re
from ipaddress import ip_network
from time import sleep
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import describe_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def route_spec_matches_route_cidr(route_spec, route):
    if route_spec.get('DestinationCidrBlock') and route.get('DestinationCidrBlock'):
        return route_spec.get('DestinationCidrBlock') == route.get('DestinationCidrBlock')
    if route_spec.get('DestinationIpv6CidrBlock') and route.get('DestinationIpv6CidrBlock'):
        return route_spec.get('DestinationIpv6CidrBlock') == route.get('DestinationIpv6CidrBlock')
    return False
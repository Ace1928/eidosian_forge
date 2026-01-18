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
def create_route_spec(connection, module, vpc_id):
    routes = module.params.get('routes')
    for route_spec in routes:
        cidr_block_type = str(type(ip_network(route_spec['dest'])))
        if 'IPv4' in cidr_block_type:
            rename_key(route_spec, 'dest', 'destination_cidr_block')
        if 'IPv6' in cidr_block_type:
            rename_key(route_spec, 'dest', 'destination_ipv6_cidr_block')
        if route_spec.get('gateway_id') and route_spec['gateway_id'].lower() == 'igw':
            igw = find_igw(connection, module, vpc_id)
            route_spec['gateway_id'] = igw
        if route_spec.get('gateway_id') and route_spec['gateway_id'].startswith('nat-'):
            rename_key(route_spec, 'gateway_id', 'nat_gateway_id')
        if route_spec.get('gateway_id') and route_spec['gateway_id'].startswith('cagw-'):
            rename_key(route_spec, 'gateway_id', 'carrier_gateway_id')
    return snake_dict_to_camel_dict(routes, capitalize_first=True)
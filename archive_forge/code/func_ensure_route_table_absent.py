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
def ensure_route_table_absent(connection, module):
    lookup = module.params.get('lookup')
    route_table_id = module.params.get('route_table_id')
    tags = module.params.get('tags')
    vpc_id = module.params.get('vpc_id')
    purge_subnets = module.params.get('purge_subnets')
    if lookup == 'tag':
        if tags is not None:
            route_table = get_route_table_by_tags(connection, module, vpc_id, tags)
        else:
            route_table = None
    elif lookup == 'id':
        route_table = get_route_table_by_id(connection, module, route_table_id)
    if route_table is None:
        return {'changed': False}
    if not module.check_mode:
        ensure_subnet_associations(connection=connection, module=module, route_table=route_table, subnets=[], purge_subnets=purge_subnets)
        disassociate_gateway(connection=connection, module=module, route_table=route_table)
        try:
            connection.delete_route_table(aws_retry=True, RouteTableId=route_table['RouteTableId'])
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg='Error deleting route table')
    return {'changed': True}
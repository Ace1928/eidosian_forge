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
def ensure_subnet_association(connection, module, vpc_id, route_table_id, subnet_id):
    filters = ansible_dict_to_boto3_filter_list({'association.subnet-id': subnet_id, 'vpc-id': vpc_id})
    try:
        route_tables = describe_route_tables_with_backoff(connection, Filters=filters)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg="Couldn't get route tables")
    for route_table in route_tables:
        if route_table.get('RouteTableId'):
            for association in route_table['Associations']:
                if association['Main']:
                    continue
                if association['SubnetId'] == subnet_id:
                    if route_table['RouteTableId'] == route_table_id:
                        return {'changed': False, 'association_id': association['RouteTableAssociationId']}
                    if module.check_mode:
                        return {'changed': True}
                    try:
                        connection.disassociate_route_table(aws_retry=True, AssociationId=association['RouteTableAssociationId'])
                    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                        module.fail_json_aws(e, msg="Couldn't disassociate subnet from route table")
    if module.check_mode:
        return {'changed': True}
    try:
        association_id = connection.associate_route_table(aws_retry=True, RouteTableId=route_table_id, SubnetId=subnet_id)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg="Couldn't associate subnet with route table")
    return {'changed': True, 'association_id': association_id}
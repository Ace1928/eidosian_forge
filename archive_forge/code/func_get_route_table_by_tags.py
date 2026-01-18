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
def get_route_table_by_tags(connection, module, vpc_id, tags):
    count = 0
    route_table = None
    filters = ansible_dict_to_boto3_filter_list({'vpc-id': vpc_id})
    try:
        route_tables = describe_route_tables_with_backoff(connection, Filters=filters)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg="Couldn't get route table")
    for table in route_tables:
        this_tags = describe_ec2_tags(connection, module, table['RouteTableId'])
        if tags_match(tags, this_tags):
            route_table = table
            count += 1
    if count > 1:
        module.fail_json(msg='Tags provided do not identify a unique route table')
    else:
        return route_table
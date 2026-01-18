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
def ensure_propagation(connection, module, route_table, propagating_vgw_ids):
    changed = False
    gateways = [gateway['GatewayId'] for gateway in route_table['PropagatingVgws']]
    vgws_to_add = set(propagating_vgw_ids) - set(gateways)
    if vgws_to_add:
        changed = True
        if not module.check_mode:
            for vgw_id in vgws_to_add:
                try:
                    connection.enable_vgw_route_propagation(aws_retry=True, RouteTableId=route_table['RouteTableId'], GatewayId=vgw_id)
                except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                    module.fail_json_aws(e, msg="Couldn't enable route propagation")
    return changed
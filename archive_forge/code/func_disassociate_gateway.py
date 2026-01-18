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
def disassociate_gateway(connection, module, route_table):
    changed = False
    associations_to_delete = [association['RouteTableAssociationId'] for association in route_table['Associations'] if not association['Main'] and association.get('GatewayId') and (association['AssociationState']['State'] in ['associated', 'associating'])]
    for association_id in associations_to_delete:
        changed = True
        if not module.check_mode:
            try:
                connection.disassociate_route_table(aws_retry=True, AssociationId=association_id)
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                module.fail_json_aws(e, msg="Couldn't disassociate gateway from route table")
    return changed
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def get_matching_igw(self, vpc_id, gateway_id=None):
    """
        Returns the internet gateway found.
            Parameters:
                vpc_id (str): VPC ID
                gateway_id (str): Internet Gateway ID, if specified
            Returns:
                igw (dict): dict of igw found, None if none found
        """
    try:
        if not gateway_id:
            filters = ansible_dict_to_boto3_filter_list({'attachment.vpc-id': vpc_id})
            igws = describe_igws_with_backoff(self._connection, Filters=filters)
        else:
            igws = describe_igws_with_backoff(self._connection, InternetGatewayIds=[gateway_id])
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        self._module.fail_json_aws(e)
    igw = None
    if len(igws) > 1:
        self._module.fail_json(msg=f'EC2 returned more than one Internet Gateway for VPC {vpc_id}, aborting')
    elif igws:
        igw = camel_dict_to_snake_dict(igws[0])
    return igw
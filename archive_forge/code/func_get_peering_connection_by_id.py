from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import add_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_peering_connection_by_id(peering_id, client, module):
    params = dict()
    params['VpcPeeringConnectionIds'] = [peering_id]
    try:
        vpc_peering_connection = client.describe_vpc_peering_connections(aws_retry=True, **params)
        return vpc_peering_connection['VpcPeeringConnections'][0]
    except is_boto3_error_code('InvalidVpcPeeringConnectionId.Malformed') as e:
        module.fail_json_aws(e, msg='Malformed connection ID')
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Error while describing peering connection by peering_id')
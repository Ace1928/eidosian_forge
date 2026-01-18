from time import sleep
from time import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_matching_tgw(self, tgw_id, description=None, skip_deleted=True):
    """search for  an existing tgw by either tgw_id or description
        :param tgw_id:  The AWS id of the transit gateway
        :param description:  The description of the transit gateway.
        :param skip_deleted: ignore deleted transit gateways
        :return dict: transit gateway object
        """
    filters = []
    if tgw_id:
        filters = ansible_dict_to_boto3_filter_list({'transit-gateway-id': tgw_id})
    try:
        response = AWSRetry.exponential_backoff()(self._connection.describe_transit_gateways)(Filters=filters)
    except (ClientError, BotoCoreError) as e:
        self._module.fail_json_aws(e)
    tgw = None
    tgws = []
    if len(response.get('TransitGateways', [])) == 1 and tgw_id:
        if response['TransitGateways'][0]['State'] != 'deleted' or not skip_deleted:
            tgws.extend(response['TransitGateways'])
    for gateway in response.get('TransitGateways', []):
        if description == gateway['Description'] and gateway['State'] != 'deleted':
            tgws.append(gateway)
    if len(tgws) > 1:
        self._module.fail_json(msg=f'EC2 returned more than one transit Gateway for description {description}, aborting')
    elif tgws:
        tgw = camel_dict_to_snake_dict(tgws[0], ignore_list=['Tags'])
        tgw['tags'] = boto3_tag_list_to_ansible_dict(tgws[0]['Tags'])
    return tgw
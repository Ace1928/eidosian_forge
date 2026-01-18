from time import sleep
from time import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_tgw(self, description):
    """
        Create a transit gateway and optionally wait for status to become available.

        :param description: The description of the transit gateway.
        :return dict: transit gateway object
        """
    options = dict()
    wait = self._module.params.get('wait')
    wait_timeout = self._module.params.get('wait_timeout')
    if self._module.params.get('asn'):
        options['AmazonSideAsn'] = self._module.params.get('asn')
    options['AutoAcceptSharedAttachments'] = self.enable_option_flag(self._module.params.get('auto_attach'))
    options['DefaultRouteTableAssociation'] = self.enable_option_flag(self._module.params.get('auto_associate'))
    options['DefaultRouteTablePropagation'] = self.enable_option_flag(self._module.params.get('auto_propagate'))
    options['VpnEcmpSupport'] = self.enable_option_flag(self._module.params.get('vpn_ecmp_support'))
    options['DnsSupport'] = self.enable_option_flag(self._module.params.get('dns_support'))
    try:
        response = self._connection.create_transit_gateway(Description=description, Options=options)
    except (ClientError, BotoCoreError) as e:
        self._module.fail_json_aws(e)
    tgw_id = response['TransitGateway']['TransitGatewayId']
    if wait:
        result = self.wait_for_status(wait_timeout=wait_timeout, tgw_id=tgw_id, status='available')
    else:
        result = self.get_matching_tgw(tgw_id=tgw_id)
    self._results['msg'] = f' Transit gateway {result['transit_gateway_id']} created'
    return result
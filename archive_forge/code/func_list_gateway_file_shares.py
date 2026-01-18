from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def list_gateway_file_shares(self, gateway):
    try:
        response = self.client.list_file_shares(GatewayARN=gateway['gateway_arn'], Limit=100)
        gateway['file_shares'] = []
        marker = self._read_gateway_fileshare_response(gateway['file_shares'], response)
        while marker is not None:
            response = self.client.list_file_shares(GatewayARN=gateway['gateway_arn'], Marker=marker, Limit=100)
            marker = self._read_gateway_fileshare_response(gateway['file_shares'], response)
    except (BotoCoreError, ClientError) as e:
        self.module.fail_json_aws(e, msg="Couldn't list gateway file shares")
from time import sleep
from time import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def ensure_tgw_present(self, tgw_id=None, description=None):
    """
        Will create a tgw if no match to the tgw_id or description are found
        Will update the tgw tags if matching one found but tags are not synced

        :param tgw_id:  The AWS id of the transit gateway
        :param description:  The description of the transit gateway.
        :return dict: transit gateway object
        """
    tgw = self.get_matching_tgw(tgw_id, description)
    if tgw is None:
        if self._check_mode:
            self._results['changed'] = True
            self._results['transit_gateway_id'] = None
            return self._results
        try:
            if not description:
                self._module.fail_json(msg='Failed to create Transit Gateway: description argument required')
            tgw = self.create_tgw(description)
            self._results['changed'] = True
        except (BotoCoreError, ClientError) as e:
            self._module.fail_json_aws(e, msg='Unable to create Transit Gateway')
    self._results['changed'] |= ensure_ec2_tags(self._connection, self._module, tgw['transit_gateway_id'], tags=self._module.params.get('tags'), purge_tags=self._module.params.get('purge_tags'))
    self._results['transit_gateway'] = self.get_matching_tgw(tgw_id=tgw['transit_gateway_id'])
    return self._results
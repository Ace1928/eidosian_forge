import time
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.arn import parse_aws_arn
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.base import BaseResourceManager
from ansible_collections.community.aws.plugins.module_utils.base import BaseWaiterFactory
from ansible_collections.community.aws.plugins.module_utils.base import Boto3Mixin
from ansible_collections.community.aws.plugins.module_utils.ec2 import BaseEc2Manager
def set_custom_stateless_actions(self, actions, purge_actions):
    if actions is None:
        return False
    new_action_list = [self._format_custom_action(a) for a in actions]
    new_action_map = self._custom_action_map(new_action_list)
    existing_action_map = self._custom_action_map(self._get_resource_value('StatelessCustomActions', []))
    if purge_actions:
        desired_action_map = dict()
    else:
        desired_action_map = deepcopy(existing_action_map)
    desired_action_map.update(new_action_map)
    if desired_action_map == existing_action_map:
        return False
    action_list = [dict(ActionName=k, ActionDefinition=v) for k, v in desired_action_map.items()]
    self._set_resource_value('StatelessCustomActions', action_list)
from copy import deepcopy
from functools import wraps
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def set_wait_timeout(self, timeout):
    if timeout is None:
        return False
    if timeout == self._wait_timeout:
        return False
    self._wait_timeout = timeout
    return True
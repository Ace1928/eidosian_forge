import traceback
from copy import deepcopy
from .ec2 import get_ec2_security_group_ids_from_names
from .elb_utils import convert_tg_name_to_arn
from .elb_utils import get_elb
from .elb_utils import get_elb_listener
from .retries import AWSRetry
from .tagging import ansible_dict_to_boto3_tag_list
from .tagging import boto3_tag_list_to_ansible_dict
from .waiters import get_waiter
def compare_listeners(self):
    """

        :return:
        """
    listeners_to_modify = []
    listeners_to_delete = []
    listeners_to_add = deepcopy(self.listeners)
    for current_listener in self.current_listeners:
        current_listener_passed_to_module = False
        for new_listener in self.listeners[:]:
            new_listener['Port'] = int(new_listener['Port'])
            if current_listener['Port'] == new_listener['Port']:
                current_listener_passed_to_module = True
                listeners_to_add.remove(new_listener)
                modified_listener = self._compare_listener(current_listener, new_listener)
                if modified_listener:
                    modified_listener['Port'] = current_listener['Port']
                    modified_listener['ListenerArn'] = current_listener['ListenerArn']
                    listeners_to_modify.append(modified_listener)
                break
        if not current_listener_passed_to_module and self.purge_listeners:
            listeners_to_delete.append(current_listener['ListenerArn'])
    return (listeners_to_add, listeners_to_modify, listeners_to_delete)
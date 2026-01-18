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
def modify_ip_address_type(self, ip_addr_type):
    """
        Modify ELB ip address type
        :return:
        """
    if ip_addr_type is None:
        return
    if self.elb_ip_addr_type == ip_addr_type:
        return
    try:
        AWSRetry.jittered_backoff()(self.connection.set_ip_address_type)(LoadBalancerArn=self.elb['LoadBalancerArn'], IpAddressType=ip_addr_type)
    except (BotoCoreError, ClientError) as e:
        self.module.fail_json_aws(e)
    self.changed = True
    self.wait_for_ip_type(self.elb['LoadBalancerArn'], ip_addr_type)
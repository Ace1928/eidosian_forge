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
def create_elb(self):
    """
        Create a load balancer
        :return:
        """
    params = self._elb_create_params()
    try:
        self.elb = AWSRetry.jittered_backoff()(self.connection.create_load_balancer)(**params)['LoadBalancers'][0]
        self.changed = True
        self.new_load_balancer = True
    except (BotoCoreError, ClientError) as e:
        self.module.fail_json_aws(e)
    self.wait_for_status(self.elb['LoadBalancerArn'])
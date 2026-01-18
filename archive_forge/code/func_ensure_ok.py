from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import get_ec2_security_group_ids_from_names
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def ensure_ok(self):
    """Create the ELB"""
    if not self.elb:
        try:
            self._create_elb()
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed to create load balancer')
        try:
            self.elb_attributes = self._get_elb_attributes()
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            self.module.fail_json_aws(e, msg='Unable to describe load balancer attributes')
        self._wait_created()
    elif self._check_scheme():
        self.ensure_gone()
        self._wait_gone(True)
        try:
            self._create_elb()
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg='Failed to recreate load balancer')
        try:
            self.elb_attributes = self._get_elb_attributes()
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            self.module.fail_json_aws(e, msg='Unable to describe load balancer attributes')
    else:
        self._set_subnets()
        self._set_zones()
        self._set_security_groups()
        self._set_elb_listeners()
        self._set_tags()
    self._set_health_check()
    self._set_elb_attributes()
    self._set_backend_policies()
    self._set_stickiness_policies()
    self._set_instance_ids()
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.elb_utils import get_elb_listener_rules
from ansible_collections.amazon.aws.plugins.module_utils.elbv2 import ApplicationLoadBalancer
from ansible_collections.amazon.aws.plugins.module_utils.elbv2 import ELBListener
from ansible_collections.amazon.aws.plugins.module_utils.elbv2 import ELBListenerRule
from ansible_collections.amazon.aws.plugins.module_utils.elbv2 import ELBListenerRules
from ansible_collections.amazon.aws.plugins.module_utils.elbv2 import ELBListeners
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
@AWSRetry.jittered_backoff()
def describe_sgs_with_backoff(connection, **params):
    paginator = connection.get_paginator('describe_security_groups')
    return paginator.paginate(**params).build_full_result()['SecurityGroups']
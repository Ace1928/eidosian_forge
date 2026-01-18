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
def find_default_sg(connection, module, vpc_id):
    """
    Finds the default security group for the given VPC ID.
    """
    filters = ansible_dict_to_boto3_filter_list({'vpc-id': vpc_id, 'group-name': 'default'})
    try:
        sg = describe_sgs_with_backoff(connection, Filters=filters)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg=f'No default security group found for VPC {vpc_id}')
    if len(sg) == 1:
        return sg[0]['GroupId']
    elif len(sg) == 0:
        module.fail_json(msg=f'No default security group found for VPC {vpc_id}')
    else:
        module.fail_json(msg=f'Multiple security groups named "default" found for VPC {vpc_id}')
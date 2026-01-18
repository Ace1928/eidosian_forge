from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def lb_instance_health(connection, load_balancer_name, instances, state):
    instance_states = connection.describe_instance_health(LoadBalancerName=load_balancer_name, Instances=instances).get('InstanceStates', [])
    instate = [instance['InstanceId'] for instance in instance_states if instance['State'] == state]
    return (instate, len(instate))
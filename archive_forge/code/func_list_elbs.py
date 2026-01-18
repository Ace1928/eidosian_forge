from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def list_elbs(connection, load_balancer_names):
    results = []
    if not load_balancer_names:
        for lb in get_all_lb(connection):
            results.append(describe_elb(connection, lb))
    for load_balancer_name in load_balancer_names:
        lb = get_lb(connection, load_balancer_name)
        if not lb:
            continue
        results.append(describe_elb(connection, lb))
    return results
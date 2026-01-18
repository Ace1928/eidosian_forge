import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
@AWSRetry.jittered_backoff(retries=5, delay=5)
def list_nodes_with_backoff(client, cluster_arn):
    paginator = client.get_paginator('list_nodes')
    return paginator.paginate(ClusterArn=cluster_arn).build_full_result()
import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
@AWSRetry.jittered_backoff(retries=5, delay=5)
def list_clusters_with_backoff(client, cluster_name):
    paginator = client.get_paginator('list_clusters')
    return paginator.paginate(ClusterNameFilter=cluster_name).build_full_result()
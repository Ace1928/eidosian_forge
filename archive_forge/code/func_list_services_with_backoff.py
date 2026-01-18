from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
@AWSRetry.jittered_backoff(retries=5, delay=5, backoff=2.0)
def list_services_with_backoff(self, **kwargs):
    paginator = self.ecs.get_paginator('list_services')
    try:
        return paginator.paginate(**kwargs).build_full_result()
    except is_boto3_error_code('ClusterNotFoundException') as e:
        self.module.fail_json_aws(e, 'Could not find cluster to list services')
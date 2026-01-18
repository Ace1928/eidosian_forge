from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
@AWSRetry.jittered_backoff(**backoff_params)
def replication_subnet_group_create(connection, **params):
    """creates the replication subnet group"""
    return connection.create_replication_subnet_group(**params)
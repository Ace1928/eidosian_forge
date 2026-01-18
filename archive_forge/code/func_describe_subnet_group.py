from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
@AWSRetry.jittered_backoff(**backoff_params)
def describe_subnet_group(connection, subnet_group):
    """checks if instance exists"""
    try:
        subnet_group_filter = dict(Name='replication-subnet-group-id', Values=[subnet_group])
        return connection.describe_replication_subnet_groups(Filters=[subnet_group_filter])
    except botocore.exceptions.ClientError:
        return {'ReplicationSubnetGroups': []}
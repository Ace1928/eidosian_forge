from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
@AWSRetry.exponential_backoff(catch_extra_error_codes=['InvalidPlacementGroup.Unknown'])
def get_placement_group_information(connection, name):
    """
    Retrieve information about a placement group.
    """
    response = connection.describe_placement_groups(GroupNames=[name])
    placement_group = response['PlacementGroups'][0]
    return {'name': placement_group['GroupName'], 'state': placement_group['State'], 'strategy': placement_group['Strategy']}
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_current_ttl_state(c, table_name):
    """Fetch the state dict for a table."""
    current_state = c.describe_time_to_live(TableName=table_name)
    return current_state.get('TimeToLiveDescription')
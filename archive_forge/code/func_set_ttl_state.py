from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def set_ttl_state(c, table_name, state, attribute_name):
    """Set our specification. Returns the update_time_to_live specification dict,
    which is different than the describe_* call."""
    is_enabled = False
    if state.lower() == 'enable':
        is_enabled = True
    ret = c.update_time_to_live(TableName=table_name, TimeToLiveSpecification={'Enabled': is_enabled, 'AttributeName': attribute_name})
    return ret.get('TimeToLiveSpecification')
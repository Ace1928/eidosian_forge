from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_for_ecs_arn(self, ecs_arn, skip_value=False):
    """
        Returns list of attribute dicts ready to be passed to boto3
        attributes put/delete methods.
        """
    return [self._setup_attr_obj(ecs_arn, skip_value=skip_value, **attr) for attr in self.attributes]
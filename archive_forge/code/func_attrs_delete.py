from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def attrs_delete(self, attrs):
    """Deletes attributes from ECS container instance."""
    try:
        self.ecs.delete_attributes(cluster=self.cluster, attributes=attrs.get_for_ecs_arn(self.ecs_arn, skip_value=True))
    except ClientError as e:
        self.module.fail_json(msg=str(e))
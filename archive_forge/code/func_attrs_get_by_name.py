from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def attrs_get_by_name(self, attrs):
    """
        Returns EcsAttributes object containing attributes from ECS container instance with names
        matching to attrs.attributes (EcsAttributes Object).
        """
    attr_objs = [{'targetType': 'container-instance', 'attributeName': attr['name']} for attr in attrs]
    try:
        matched_ecs_targets = [attr_found for attr_obj in attr_objs for attr_found in self.ecs.list_attributes(cluster=self.cluster, **attr_obj)['attributes']]
    except ClientError as e:
        self.module.fail_json(msg=f"Can't connect to the cluster - {str(e)}")
    matched_objs = [target for target in matched_ecs_targets if target['targetId'] == self.ecs_arn]
    results = [{'name': match['name'], 'value': match.get('value', None)} for match in matched_objs]
    return EcsAttributes(self.module, results)
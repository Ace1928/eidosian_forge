from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def describe_task_definitions(self, family):
    data = {'taskDefinitionArns': [], 'nextToken': None}

    def fetch():
        params = {'familyPrefix': family}
        if data['nextToken']:
            params['nextToken'] = data['nextToken']
        result = self.ecs.list_task_definitions(**params)
        data['taskDefinitionArns'] += result['taskDefinitionArns']
        data['nextToken'] = result.get('nextToken', None)
        return data['nextToken'] is not None
    while fetch():
        pass
    return list(sorted([self.ecs.describe_task_definition(taskDefinition=arn)['taskDefinition'] for arn in data['taskDefinitionArns']], key=lambda td: td['revision']))
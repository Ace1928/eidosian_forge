from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def extract_service_from(self, service):
    if 'deployments' in service:
        for d in service['deployments']:
            if 'createdAt' in d:
                d['createdAt'] = str(d['createdAt'])
            if 'updatedAt' in d:
                d['updatedAt'] = str(d['updatedAt'])
    if 'events' in service:
        if not self.module.params['events']:
            del service['events']
        else:
            for e in service['events']:
                if 'createdAt' in e:
                    e['createdAt'] = str(e['createdAt'])
    return service
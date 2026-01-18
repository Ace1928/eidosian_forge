from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.batch import set_api_params
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_current_job_queue(module, client):
    try:
        environments = client.describe_job_queues(jobQueues=[module.params['job_queue_name']])
        return environments['jobQueues'][0] if len(environments['jobQueues']) > 0 else None
    except ClientError:
        return None
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.batch import set_api_params
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def remove_job_queue(module, client):
    """
    Remove a Batch job queue

    :param module:
    :param client:
    :return:
    """
    changed = False
    api_params = {'jobQueue': module.params['job_queue_name']}
    try:
        if not module.check_mode:
            client.delete_job_queue(**api_params)
        changed = True
    except (BotoCoreError, ClientError) as e:
        module.fail_json_aws(e, msg='Error removing job queue')
    return changed
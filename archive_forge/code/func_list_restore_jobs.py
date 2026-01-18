from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
def list_restore_jobs(connection, module, request_args):
    try:
        response = _list_restore_jobs(connection, **request_args)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to list restore jobs')
    return [camel_dict_to_snake_dict(restore_job) for restore_job in response['RestoreJobs']]
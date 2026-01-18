import copy
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_aws_account_info
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_or_update_glue_job(connection, module, glue_job):
    """
    Create or update an AWS Glue job

    :param connection: AWS boto3 glue connection
    :param module: Ansible module
    :param glue_job: a dict of AWS Glue job parameters or None
    :return:
    """
    changed = False
    params = dict()
    params['Name'] = module.params.get('name')
    params['Role'] = module.params.get('role')
    if module.params.get('allocated_capacity') is not None:
        params['AllocatedCapacity'] = module.params.get('allocated_capacity')
    if module.params.get('command_script_location') is not None:
        params['Command'] = {'Name': module.params.get('command_name'), 'ScriptLocation': module.params.get('command_script_location')}
        if module.params.get('command_python_version') is not None:
            params['Command']['PythonVersion'] = module.params.get('command_python_version')
    if module.params.get('connections') is not None:
        params['Connections'] = {'Connections': module.params.get('connections')}
    if module.params.get('default_arguments') is not None:
        params['DefaultArguments'] = module.params.get('default_arguments')
    if module.params.get('description') is not None:
        params['Description'] = module.params.get('description')
    if module.params.get('glue_version') is not None:
        params['GlueVersion'] = module.params.get('glue_version')
    if module.params.get('max_concurrent_runs') is not None:
        params['ExecutionProperty'] = {'MaxConcurrentRuns': module.params.get('max_concurrent_runs')}
    if module.params.get('max_retries') is not None:
        params['MaxRetries'] = module.params.get('max_retries')
    if module.params.get('timeout') is not None:
        params['Timeout'] = module.params.get('timeout')
    if module.params.get('glue_version') is not None:
        params['GlueVersion'] = module.params.get('glue_version')
    if module.params.get('worker_type') is not None:
        params['WorkerType'] = module.params.get('worker_type')
    if module.params.get('number_of_workers') is not None:
        params['NumberOfWorkers'] = module.params.get('number_of_workers')
    if glue_job:
        if _compare_glue_job_params(params, glue_job):
            try:
                update_params = {'JobName': params['Name'], 'JobUpdate': copy.deepcopy(params)}
                del update_params['JobUpdate']['Name']
                if not module.check_mode:
                    connection.update_job(aws_retry=True, **update_params)
                changed = True
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                module.fail_json_aws(e)
    else:
        try:
            if not module.check_mode:
                connection.create_job(aws_retry=True, **params)
            changed = True
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e)
    glue_job = _get_glue_job(connection, module, params['Name'])
    changed |= ensure_tags(connection, module, glue_job)
    module.exit_json(changed=changed, **camel_dict_to_snake_dict(glue_job or {}, ignore_list=['DefaultArguments']))
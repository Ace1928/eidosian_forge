from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def start_execution(module, sfn_client):
    """
    start_execution uses execution name to determine if a previous execution already exists.
    If an execution by the provided name exists, call client.start_execution will not be called.
    """
    state_machine_arn = module.params.get('state_machine_arn')
    name = module.params.get('name')
    execution_input = module.params.get('execution_input')
    try:
        page_iterators = sfn_client.get_paginator('list_executions').paginate(stateMachineArn=state_machine_arn)
        for execution in page_iterators.build_full_result()['executions']:
            if name == execution['name']:
                check_mode(module, msg='State machine execution already exists.', changed=False)
                module.exit_json(changed=False)
        check_mode(module, msg='State machine execution would be started.', changed=True)
        res_execution = sfn_client.start_execution(stateMachineArn=state_machine_arn, name=name, input=execution_input)
    except is_boto3_error_code('ExecutionAlreadyExists'):
        module.exit_json(changed=False)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to start execution.')
    module.exit_json(changed=True, **camel_dict_to_snake_dict(res_execution))
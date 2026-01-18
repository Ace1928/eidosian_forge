import re
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.exceptions import AnsibleAWSError
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
def lambda_alias(module_params, client, check_mode):
    """
    Adds, updates or deletes lambda function aliases.

    :param module_params: AnsibleAWSModule parameters
    :param client: (wrapped) boto3 lambda client
    :return dict:
    """
    results = dict()
    changed = False
    current_state = 'absent'
    state = module_params['state']
    facts = get_lambda_alias(module_params, client)
    if facts:
        current_state = 'present'
    if state == 'present':
        if current_state == 'present':
            snake_facts = camel_dict_to_snake_dict(facts)
            alias_params = ('function_version', 'description')
            for param in alias_params:
                if module_params.get(param) is None:
                    continue
                if module_params.get(param) != snake_facts.get(param):
                    changed = True
                    break
            if changed:
                api_params = set_api_params(module_params, ('function_name', 'name'))
                api_params.update(set_api_params(module_params, alias_params))
                if not check_mode:
                    try:
                        results = client.update_alias(aws_retry=True, **api_params)
                    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                        raise LambdaAnsibleAWSError('Error updating function alias', exception=e)
        else:
            api_params = set_api_params(module_params, ('function_name', 'name', 'function_version', 'description'))
            try:
                if not check_mode:
                    results = client.create_alias(aws_retry=True, **api_params)
                changed = True
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                raise LambdaAnsibleAWSError('Error creating function alias', exception=e)
    elif current_state == 'present':
        api_params = set_api_params(module_params, ('function_name', 'name'))
        try:
            if not check_mode:
                results = client.delete_alias(aws_retry=True, **api_params)
            changed = True
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            raise LambdaAnsibleAWSError('Error deleting function alias', exception=e)
    return dict(changed=changed, **dict(results or facts or {}))
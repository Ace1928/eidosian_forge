import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.base import BaseWaiterFactory
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_update_parameter(client, module):
    changed = False
    existing_parameter = None
    response = {}
    args = dict(Name=module.params.get('name'), Type=module.params.get('string_type'), Tier=module.params.get('tier'))
    if module.params.get('overwrite_value') in ('always', 'changed'):
        args.update(Overwrite=True)
    else:
        args.update(Overwrite=False)
    if module.params.get('value') is not None:
        args.update(Value=module.params.get('value'))
    if module.params.get('description'):
        args.update(Description=module.params.get('description'))
    if module.params.get('string_type') == 'SecureString':
        args.update(KeyId=module.params.get('key_id'))
    try:
        existing_parameter = client.get_parameter(aws_retry=True, Name=args['Name'], WithDecryption=True)
    except botocore.exceptions.ClientError:
        pass
    except botocore.exceptions.BotoCoreError as e:
        module.fail_json_aws(e, msg='fetching parameter')
    if existing_parameter:
        original_version = existing_parameter['Parameter']['Version']
        if 'Value' not in args:
            args['Value'] = existing_parameter['Parameter']['Value']
        if module.params.get('overwrite_value') == 'always':
            changed, response = update_parameter(client, module, **args)
        elif module.params.get('overwrite_value') == 'changed':
            if existing_parameter['Parameter']['Type'] != args['Type']:
                changed, response = update_parameter(client, module, **args)
            elif existing_parameter['Parameter']['Value'] != args['Value']:
                changed, response = update_parameter(client, module, **args)
            elif args.get('Description'):
                try:
                    describe_existing_parameter = describe_parameter(client, module, ParameterFilters=[{'Key': 'Name', 'Values': [args['Name']]}])
                except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                    module.fail_json_aws(e, msg='getting description value')
                if describe_existing_parameter.get('Description') != args['Description']:
                    changed, response = update_parameter(client, module, **args)
        if changed:
            _wait_updated(client, module, module.params.get('name'), original_version)
        if module.params.get('overwrite_value') != 'never':
            tags_changed, tags_response = update_parameter_tags(client, module, existing_parameter['Parameter']['Name'], module.params.get('tags'))
            changed = changed or tags_changed
            if tags_response:
                response['tag_updates'] = tags_response
    else:
        if module.params.get('tags'):
            args.update(Tags=ansible_dict_to_boto3_tag_list(module.params.get('tags')))
            args.update(Overwrite=False)
        changed, response = update_parameter(client, module, **args)
        _wait_exists(client, module, module.params.get('name'))
    return (changed, response)
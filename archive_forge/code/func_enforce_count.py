import time
import uuid
from collections import namedtuple
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.arn import validate_aws_arn
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import get_ec2_security_group_ids_from_names
from ansible_collections.amazon.aws.plugins.module_utils.exceptions import AnsibleAWSError
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.tower import tower_callback_script
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def enforce_count(existing_matches, module, desired_module_state):
    exact_count = module.params.get('exact_count')
    try:
        current_count = len(existing_matches)
        if current_count == exact_count:
            module.exit_json(changed=False, instances=[pretty_instance(i) for i in existing_matches], instance_ids=[i['InstanceId'] for i in existing_matches], msg=f'{exact_count} instances already running, nothing to do.')
        elif current_count < exact_count:
            try:
                ensure_present(existing_matches=existing_matches, desired_module_state=desired_module_state, current_count=current_count)
            except botocore.exceptions.ClientError as e:
                module.fail_json(e, msg='Unable to launch instances')
        elif current_count > exact_count:
            to_terminate = current_count - exact_count
            existing_matches = sorted(existing_matches, key=lambda inst: inst['LaunchTime'])
            all_instance_ids = [x['InstanceId'] for x in existing_matches]
            terminate_ids = all_instance_ids[0:to_terminate]
            if module.check_mode:
                module.exit_json(changed=True, terminated_ids=terminate_ids, instance_ids=all_instance_ids, msg=f'Would have terminated following instances if not in check mode {terminate_ids}')
            try:
                client.terminate_instances(aws_retry=True, InstanceIds=terminate_ids)
                await_instances(terminate_ids, desired_module_state='terminated', force_wait=True)
            except is_boto3_error_code('InvalidInstanceID.NotFound'):
                pass
            except botocore.exceptions.ClientError as e:
                module.fail_json(e, msg='Unable to terminate instances')
            module.exit_json(changed=True, msg='Successfully terminated instances.', terminated_ids=terminate_ids, instance_ids=all_instance_ids, instances=existing_matches)
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, msg='Failed to enforce instance count')
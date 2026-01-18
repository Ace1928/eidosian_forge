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
def ensure_present(existing_matches, desired_module_state, current_count=None):
    tags = dict(module.params.get('tags') or {})
    name = module.params.get('name')
    if name:
        tags['Name'] = name
    try:
        instance_spec = build_run_instance_spec(module.params, current_count)
        if module.check_mode:
            if existing_matches:
                instance_ids = [x['InstanceId'] for x in existing_matches]
                module.exit_json(changed=True, instance_ids=instance_ids, instances=existing_matches, spec=instance_spec, msg='Would have launched instances if not in check_mode.')
            else:
                module.exit_json(changed=True, spec=instance_spec, msg='Would have launched instances if not in check_mode.')
        instance_response = run_instances(**instance_spec)
        instances = instance_response['Instances']
        instance_ids = [i['InstanceId'] for i in instances]
        await_instances(instance_ids, desired_module_state='present', force_wait=True)
        for ins in instances:
            try:
                AWSRetry.jittered_backoff(catch_extra_error_codes=['InvalidInstanceID.NotFound'])(client.describe_instance_status)(InstanceIds=[ins['InstanceId']], IncludeAllInstances=True)
            except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
                module.fail_json_aws(e, msg='Failed to fetch status of new EC2 instance')
            changes = diff_instance_and_params(ins, module.params, skip=['UserData', 'EbsOptimized'])
            for c in changes:
                try:
                    client.modify_instance_attribute(aws_retry=True, **c)
                except botocore.exceptions.ClientError as e:
                    module.fail_json_aws(e, msg=f'Could not apply change {str(c)} to new instance.')
        if existing_matches:
            all_instance_ids = [x['InstanceId'] for x in existing_matches] + instance_ids
        if not module.params.get('wait'):
            if existing_matches:
                module.exit_json(changed=True, changed_ids=instance_ids, instance_ids=all_instance_ids, spec=instance_spec)
            else:
                module.exit_json(changed=True, instance_ids=instance_ids, spec=instance_spec)
        await_instances(instance_ids, desired_module_state=desired_module_state)
        instances = find_instances(ids=instance_ids)
        if existing_matches:
            all_instances = existing_matches + instances
            module.exit_json(changed=True, changed_ids=instance_ids, instance_ids=all_instance_ids, instances=[pretty_instance(i) for i in all_instances], spec=instance_spec)
        else:
            module.exit_json(changed=True, instance_ids=instance_ids, instances=[pretty_instance(i) for i in instances], spec=instance_spec)
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, msg='Failed to create new EC2 instance')
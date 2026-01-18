import json
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_or_update_sqs_queue(client, module):
    is_fifo = module.params.get('queue_type') == 'fifo'
    kms_master_key_id = module.params.get('kms_master_key_id')
    queue_name = get_queue_name(module, is_fifo)
    result = dict(name=queue_name, region=module.params.get('region'), changed=False)
    queue_url = get_queue_url(client, queue_name)
    result['queue_url'] = queue_url
    create_attributes = {}
    if not queue_url:
        if is_fifo:
            create_attributes['FifoQueue'] = 'True'
        if kms_master_key_id:
            create_attributes['KmsMasterKeyId'] = kms_master_key_id
        result['changed'] = True
        if module.check_mode:
            return result
        queue_url = client.create_queue(QueueName=queue_name, Attributes=create_attributes, aws_retry=True)['QueueUrl']
    changed, arn = update_sqs_queue(module, client, queue_url)
    result['changed'] |= changed
    result['queue_arn'] = arn
    changed, tags = update_tags(client, queue_url, module)
    result['changed'] |= changed
    result['tags'] = tags
    result.update(describe_queue(client, queue_url))
    COMPATABILITY_KEYS = dict(delay_seconds='delivery_delay', receive_message_wait_time_seconds='receive_message_wait_time', visibility_timeout='default_visibility_timeout', kms_data_key_reuse_period_seconds='kms_data_key_reuse_period')
    for key in list(result.keys()):
        return_name = COMPATABILITY_KEYS.get(key)
        if return_name:
            result[return_name] = result.get(key)
    return result
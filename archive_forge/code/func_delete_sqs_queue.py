import json
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def delete_sqs_queue(client, module):
    is_fifo = module.params.get('queue_type') == 'fifo'
    queue_name = get_queue_name(module, is_fifo)
    result = dict(name=queue_name, region=module.params.get('region'), changed=False)
    queue_url = get_queue_url(client, queue_name)
    if not queue_url:
        return result
    result['changed'] = bool(queue_url)
    if not module.check_mode:
        AWSRetry.jittered_backoff()(client.delete_queue)(QueueUrl=queue_url)
    return result
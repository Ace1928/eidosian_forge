import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_target_group(connection, module, retry_missing=False):
    extra_codes = ['TargetGroupNotFound'] if retry_missing else []
    try:
        target_group_paginator = connection.get_paginator('describe_target_groups').paginate(Names=[module.params.get('name')])
        jittered_retry = AWSRetry.jittered_backoff(retries=10, catch_extra_error_codes=extra_codes)
        result = jittered_retry(target_group_paginator.build_full_result)()
    except is_boto3_error_code('TargetGroupNotFound'):
        return None
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg="Couldn't get target group")
    return result['TargetGroups'][0]
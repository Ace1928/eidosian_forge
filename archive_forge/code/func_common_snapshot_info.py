from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def common_snapshot_info(module, conn, method, prefix, params):
    paginator = conn.get_paginator(method)
    try:
        results = paginator.paginate(**params).build_full_result()[f'{prefix}s']
    except is_boto3_error_code(f'{prefix}NotFound'):
        results = []
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, 'trying to get snapshot information')
    for snapshot in results:
        try:
            if snapshot['SnapshotType'] != 'shared':
                snapshot['Tags'] = boto3_tag_list_to_ansible_dict(conn.list_tags_for_resource(ResourceName=snapshot[f'{prefix}Arn'], aws_retry=True)['TagList'])
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            snapshot_name = snapshot[f'{prefix}Identifier']
            module.fail_json_aws(e, f"Couldn't get tags for snapshot {snapshot_name}")
    return [camel_dict_to_snake_dict(snapshot, ignore_list=['Tags']) for snapshot in results]
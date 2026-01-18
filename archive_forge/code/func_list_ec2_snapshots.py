from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def list_ec2_snapshots(connection, module, request_args):
    try:
        snapshots = get_snapshots(connection, module, request_args)
    except ClientError as e:
        module.fail_json_aws(e, msg='Failed to describe snapshots')
    result = {}
    for snapshot in snapshots['Snapshots']:
        snapshot_id = snapshot.get('SnapshotId')
        create_vol_permission = _describe_snapshot_attribute(module, connection, snapshot_id)
        snapshot['CreateVolumePermissions'] = create_vol_permission
    snaked_snapshots = []
    for snapshot in snapshots['Snapshots']:
        snaked_snapshots.append(camel_dict_to_snake_dict(snapshot))
    for snapshot in snaked_snapshots:
        if 'tags' in snapshot:
            snapshot['tags'] = boto3_tag_list_to_ansible_dict(snapshot['tags'], 'key', 'value')
    result['snapshots'] = snaked_snapshots
    if snapshots.get('NextToken'):
        result.update(camel_dict_to_snake_dict({'NextTokenId': snapshots.get('NextToken')}))
    return result
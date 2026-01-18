import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import add_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
@staticmethod
def defer_purge_snapshots(image):

    def purge_snapshots(connection):
        try:
            for mapping in image.get('BlockDeviceMappings') or []:
                snapshot_id = mapping.get('Ebs', {}).get('SnapshotId')
                if snapshot_id is None:
                    continue
                connection.delete_snapshot(aws_retry=True, SnapshotId=snapshot_id)
                yield snapshot_id
        except is_boto3_error_code('InvalidSnapshot.NotFound'):
            pass
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            raise Ec2AmiFailure('Failed to delete snapshot.', e)
    return purge_snapshots
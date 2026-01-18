from time import sleep
from time import time as timestamp
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def delete_mount_targets(self, file_system_id):
    """
        Removes mount targets by EFS id
        """
    wait_for(lambda: len(self.get_mount_targets_in_state(file_system_id, self.STATE_CREATING)), 0)
    targets = self.get_mount_targets_in_state(file_system_id, self.STATE_AVAILABLE)
    for target in targets:
        self.connection.delete_mount_target(MountTargetId=target['MountTargetId'])
    wait_for(lambda: len(self.get_mount_targets_in_state(file_system_id, self.STATE_DELETING)), 0)
    return len(targets) > 0
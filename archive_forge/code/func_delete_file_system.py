from time import sleep
from time import time as timestamp
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def delete_file_system(self, name, file_system_id=None):
    """
        Removes EFS instance by id/name
        """
    result = False
    state = self.get_file_system_state(name, file_system_id)
    if state in [self.STATE_CREATING, self.STATE_AVAILABLE]:
        wait_for(lambda: self.get_file_system_state(name), self.STATE_AVAILABLE)
        if not file_system_id:
            file_system_id = self.get_file_system_id(name)
        self.delete_mount_targets(file_system_id)
        self.connection.delete_file_system(FileSystemId=file_system_id)
        result = True
    if self.wait:
        wait_for(lambda: self.get_file_system_state(name), self.STATE_DELETED, self.wait_timeout)
    return result
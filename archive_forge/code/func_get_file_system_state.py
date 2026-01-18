from time import sleep
from time import time as timestamp
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_file_system_state(self, name, file_system_id=None):
    """
        Returns state of filesystem by EFS id/name
        """
    info = first_or_default(iterate_all('FileSystems', self.connection.describe_file_systems, CreationToken=name, FileSystemId=file_system_id))
    return info and info['LifeCycleState'] or self.STATE_DELETED
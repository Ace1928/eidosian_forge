from time import sleep
from time import time as timestamp
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_mount_targets(self, **kwargs):
    """
        Returns mount targets for selected instance of EFS
        """
    targets = iterate_all('MountTargets', self.connection.describe_mount_targets, **kwargs)
    for target in targets:
        if target['LifeCycleState'] == self.STATE_AVAILABLE:
            target['SecurityGroups'] = list(self.get_security_groups(MountTargetId=target['MountTargetId']))
        else:
            target['SecurityGroups'] = []
        yield target
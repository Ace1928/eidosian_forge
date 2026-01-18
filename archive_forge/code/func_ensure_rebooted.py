from time import sleep
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def ensure_rebooted(self):
    """Ensure cache cluster is gone or delete it if not"""
    self.reboot()
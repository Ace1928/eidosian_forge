from time import sleep
from time import time as timestamp
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_file_system(self, name, performance_mode, encrypt, kms_key_id, throughput_mode, provisioned_throughput_in_mibps):
    """
        Creates new filesystem with selected name
        """
    changed = False
    state = self.get_file_system_state(name)
    params = {}
    params['CreationToken'] = name
    params['PerformanceMode'] = performance_mode
    if encrypt:
        params['Encrypted'] = encrypt
    if kms_key_id is not None:
        params['KmsKeyId'] = kms_key_id
    if throughput_mode:
        params['ThroughputMode'] = throughput_mode
    if provisioned_throughput_in_mibps:
        params['ProvisionedThroughputInMibps'] = provisioned_throughput_in_mibps
    if state in [self.STATE_DELETING, self.STATE_DELETED]:
        wait_for(lambda: self.get_file_system_state(name), self.STATE_DELETED)
        try:
            self.connection.create_file_system(**params)
            changed = True
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            self.module.fail_json_aws(e, msg='Unable to create file system.')
    wait_for(lambda: self.get_file_system_state(name), self.STATE_AVAILABLE, self.wait_timeout)
    return changed
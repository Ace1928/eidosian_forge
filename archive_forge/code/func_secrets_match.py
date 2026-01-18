import json
from traceback import format_exc
from ansible.module_utils._text import to_bytes
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def secrets_match(self, desired_secret, current_secret):
    """Compare secrets except tags and rotation

        Args:
            desired_secret: camel dict representation of the desired secret state.
            current_secret: secret reference as returned by the secretsmanager api.

        Returns: bool
        """
    if desired_secret.description != current_secret.get('Description', ''):
        return False
    if desired_secret.kms_key_id != current_secret.get('KmsKeyId'):
        return False
    current_secret_value = self.client.get_secret_value(SecretId=current_secret.get('Name'))
    if desired_secret.secret_type == 'SecretBinary':
        desired_value = to_bytes(desired_secret.secret)
    else:
        desired_value = desired_secret.secret
    if desired_value != current_secret_value.get(desired_secret.secret_type):
        return False
    return True
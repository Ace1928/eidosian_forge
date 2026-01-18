from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def get_kms_key_aliases(module, client, keyId):
    """
    get list of key aliases

    module : AnsibleAWSModule object
    client : boto3 client connection object for kms
    keyId : keyId to get aliases for
    """
    try:
        key_resp = client.list_aliases(KeyId=keyId)
    except (BotoCoreError, ClientError):
        return []
    return key_resp['Aliases']
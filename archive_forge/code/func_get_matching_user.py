import secrets
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def get_matching_user(conn, module, broker_id, username):
    try:
        response = conn.describe_user(BrokerId=broker_id, Username=username)
    except is_boto3_error_code('NotFoundException'):
        return None
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg="Couldn't get user details")
    return response
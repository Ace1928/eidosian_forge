from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def delete_password_policy(self, policy):
    try:
        results = policy.delete()
    except is_boto3_error_code('NoSuchEntity'):
        self.module.exit_json(changed=False, task_status={'IAM': "Couldn't find IAM Password Policy"})
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        self.module.fail_json_aws(e, msg="Couldn't delete IAM Password Policy")
    return camel_dict_to_snake_dict(results)
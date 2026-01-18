import copy
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_aws_account_info
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def delete_glue_job(connection, module, glue_job):
    """
    Delete an AWS Glue job

    :param connection: AWS boto3 glue connection
    :param module: Ansible module
    :param glue_job: a dict of AWS Glue job parameters or None
    :return:
    """
    changed = False
    if glue_job:
        try:
            if not module.check_mode:
                connection.delete_job(aws_retry=True, JobName=glue_job['Name'])
            changed = True
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e)
    module.exit_json(changed=changed)
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_fargate_profile(client, module, name, cluster_name):
    try:
        return client.describe_fargate_profile(clusterName=cluster_name, fargateProfileName=name)['fargateProfile']
    except is_boto3_error_code('ResourceNotFoundException'):
        return None
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, msg="Couldn't get fargate profile")
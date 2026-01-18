from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def check_profiles_status(client, module, cluster_name):
    try:
        list_profiles = client.list_fargate_profiles(clusterName=cluster_name)
        for name in list_profiles['fargateProfileNames']:
            fargate_profile = get_fargate_profile(client, module, name, cluster_name)
            if fargate_profile['status'] == 'CREATING':
                wait_until(client, module, 'fargate_profile_active', fargate_profile['fargateProfileName'], cluster_name)
            elif fargate_profile['status'] == 'DELETING':
                wait_until(client, module, 'fargate_profile_deleted', fargate_profile['fargateProfileName'], cluster_name)
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, msg="Couldn't not find EKS cluster")
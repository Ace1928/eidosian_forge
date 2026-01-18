from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_subnet_group(name, description, subnets):
    if not subnets:
        module.fail_json(msg='At least one subnet must be provided when creating a subnet group')
    if module.check_mode:
        return True
    try:
        if not description:
            description = name
        client.create_cluster_subnet_group(aws_retry=True, ClusterSubnetGroupName=name, Description=description, SubnetIds=subnets)
        return True
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to create subnet group')
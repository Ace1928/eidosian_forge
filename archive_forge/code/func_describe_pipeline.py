import copy
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def describe_pipeline(client, name, version, module):
    pipeline = {}
    try:
        if version is not None:
            pipeline = client.get_pipeline(name=name, version=version)
            return pipeline
        else:
            pipeline = client.get_pipeline(name=name)
            return pipeline
    except is_boto3_error_code('PipelineNotFoundException'):
        return pipeline
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e)
import json
import traceback
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.botocore import boto_exception
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_repository_policy(self, registry_id, name):
    try:
        res = self.ecr.get_repository_policy(repositoryName=name, **build_kwargs(registry_id))
        text = res.get('policyText')
        return text and json.loads(text)
    except is_boto3_error_code(['RepositoryNotFoundException', 'RepositoryPolicyNotFoundException']):
        return None
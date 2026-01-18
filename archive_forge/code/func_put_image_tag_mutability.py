import json
import traceback
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.botocore import boto_exception
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def put_image_tag_mutability(self, registry_id, name, new_mutability_configuration):
    repo = self.get_repository(registry_id, name)
    current_mutability_configuration = repo.get('imageTagMutability')
    if current_mutability_configuration != new_mutability_configuration:
        if not self.check_mode:
            self.ecr.put_image_tag_mutability(repositoryName=name, imageTagMutability=new_mutability_configuration, **build_kwargs(registry_id))
        else:
            self.skipped = True
        self.changed = True
    repo['imageTagMutability'] = new_mutability_configuration
    return repo
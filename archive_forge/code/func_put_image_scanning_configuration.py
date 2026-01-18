import json
import traceback
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.botocore import boto_exception
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def put_image_scanning_configuration(self, registry_id, name, scan_on_push):
    if not self.check_mode:
        if registry_id:
            scan = self.ecr.put_image_scanning_configuration(registryId=registry_id, repositoryName=name, imageScanningConfiguration={'scanOnPush': scan_on_push})
        else:
            scan = self.ecr.put_image_scanning_configuration(repositoryName=name, imageScanningConfiguration={'scanOnPush': scan_on_push})
        self.changed = True
        return scan
    else:
        self.skipped = True
        return None
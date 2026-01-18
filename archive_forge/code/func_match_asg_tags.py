import re
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def match_asg_tags(tags_to_match, asg):
    for key, value in tags_to_match.items():
        for tag in asg['Tags']:
            if key == tag['Key'] and value == tag['Value']:
                break
        else:
            return False
    return True
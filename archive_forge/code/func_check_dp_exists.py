import hashlib
import json
import time
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def check_dp_exists(client, dp_id):
    """Check if datapipeline exists

    :param object client: boto3 datapipeline client
    :param string dp_id: pipeline id
    :returns: True or False

    """
    try:
        if pipeline_description(client, dp_id):
            return True
        else:
            return False
    except DataPipelineNotFound:
        return False
import hashlib
import json
import time
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def check_dp_status(client, dp_id, status):
    """Checks if datapipeline matches states in status list

    :param object client: boto3 datapipeline client
    :param string dp_id: pipeline id
    :param list status: list of states to check against
    :returns: True or False

    """
    if not isinstance(status, list):
        raise AssertionError()
    if pipeline_field(client, dp_id, field='@pipelineState') in status:
        return True
    else:
        return False
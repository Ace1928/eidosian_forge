import hashlib
import json
import time
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def pipeline_description(client, dp_id):
    """Return pipeline description list

    :param object client: boto3 datapipeline client
    :returns: pipeline description dictionary
    :raises: DataPipelineNotFound

    """
    try:
        return client.describe_pipelines(pipelineIds=[dp_id])
    except is_boto3_error_code(['PipelineNotFoundException', 'PipelineDeletedException']):
        raise DataPipelineNotFound
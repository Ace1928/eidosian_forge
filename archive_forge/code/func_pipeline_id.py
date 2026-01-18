import hashlib
import json
import time
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def pipeline_id(client, name):
    """Return pipeline id for the given pipeline name

    :param object client: boto3 datapipeline client
    :param string name: pipeline name
    :returns: pipeline id
    :raises: DataPipelineNotFound

    """
    pipelines = client.list_pipelines()
    for dp in pipelines['pipelineIdList']:
        if dp['name'] == name:
            return dp['id']
    raise DataPipelineNotFound
import hashlib
import json
import time
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def diff_pipeline(client, module, objects, unique_id, dp_name):
    """Check if there's another pipeline with the same unique_id and if so, checks if the object needs to be updated"""
    result = {}
    changed = False
    create_dp = False
    unique_id = build_unique_id(module)
    try:
        dp_id = pipeline_id(client, dp_name)
        dp_unique_id = to_text(pipeline_field(client, dp_id, field='uniqueId'))
        if dp_unique_id != unique_id:
            changed = 'NEW_VERSION'
            create_dp = True
        else:
            dp_objects = client.get_pipeline_definition(pipelineId=dp_id)['pipelineObjects']
            if dp_objects != objects:
                changed, msg = define_pipeline(client, module, objects, dp_id)
            else:
                msg = f'Data Pipeline {dp_name} is present'
            data_pipeline = get_result(client, dp_id)
            result = {'data_pipeline': data_pipeline, 'msg': msg}
    except DataPipelineNotFound:
        create_dp = True
    return (create_dp, changed, result)
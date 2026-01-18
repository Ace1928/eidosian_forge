import hashlib
import json
import time
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def define_pipeline(client, module, objects, dp_id):
    """Puts pipeline definition"""
    dp_name = module.params.get('name')
    if pipeline_field(client, dp_id, field='@pipelineState') == 'FINISHED':
        msg = f'Data Pipeline {dp_name} is unable to be updated while in state FINISHED.'
        changed = False
    elif objects:
        parameters = module.params.get('parameters')
        values = module.params.get('values')
        try:
            client.put_pipeline_definition(pipelineId=dp_id, pipelineObjects=objects, parameterObjects=parameters, parameterValues=values)
            msg = f'Data Pipeline {dp_name} has been updated.'
            changed = True
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg=f'Failed to put the definition for pipeline {dp_name}. Check that string/reference fieldsare not empty and that the number of objects in the pipeline does not exceed maximum allowedobjects')
    else:
        changed = False
        msg = ''
    return (changed, msg)
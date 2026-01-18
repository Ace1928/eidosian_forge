import json
from datetime import datetime
from typing import Optional
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.backup import get_plan_details
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
def create_backup_plan(module: AnsibleAWSModule, client, create_params: dict) -> dict:
    """
    Creates a backup plan.

    module : AnsibleAWSModule object
    client : boto3 backup client connection object
    create_params : The boto3 backup client parameters to create a backup plan
    """
    try:
        response = client.create_backup_plan(**create_params)
    except (BotoCoreError, ClientError) as err:
        module.fail_json_aws(err, msg='Failed to create backup plan {err}')
    return response
import base64
import re
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def get_broker_info(conn, module):
    try:
        return conn.describe_broker(BrokerId=module.params['broker_id'])
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        if module.check_mode:
            return {'broker_id': module.params['broker_id']}
        module.fail_json_aws(e, msg="Couldn't get broker details.")
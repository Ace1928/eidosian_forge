import base64
import re
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def create_and_assign_config(conn, module, broker_id, cfg_id, cfg_xml_encoded):
    kwargs = {'ConfigurationId': cfg_id, 'Data': cfg_xml_encoded}
    if 'config_description' in module.params and module.params['config_description']:
        kwargs['Description'] = module.params['config_description']
    else:
        kwargs['Description'] = 'Updated through community.aws.mq_broker_config ansible module'
    try:
        c_response = conn.update_configuration(**kwargs)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg="Couldn't create new configuration revision.")
    new_config_revision = c_response['LatestRevision']['Revision']
    try:
        b_response = conn.update_broker(BrokerId=broker_id, Configuration={'Id': cfg_id, 'Revision': new_config_revision})
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg="Couldn't assign new configuration revision to broker.")
    return (c_response, b_response)
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def get_broker_id(conn, module):
    try:
        broker_name = module.params['broker_name']
        broker_id = None
        response = conn.list_brokers(MaxResults=100)
        for broker in response['BrokerSummaries']:
            if broker['BrokerName'] == broker_name:
                broker_id = broker['BrokerId']
                break
        return broker_id
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg="Couldn't list broker brokers.")
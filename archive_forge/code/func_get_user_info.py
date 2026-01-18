from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.core import AnsibleAWSModule
def get_user_info(conn, module):
    try:
        response = conn.list_users(BrokerId=module.params['broker_id'], MaxResults=module.params['max_results'])
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        if module.check_mode:
            if DEFAULTS['as_dict']:
                return {}
            return []
        module.fail_json_aws(e, msg='Failed to describe users')
    if not module.params['skip_pending_create'] and (not module.params['skip_pending_delete']):
        records = response['Users']
    else:
        records = []
        for record in response['Users']:
            if 'PendingChange' in record:
                if record['PendingChange'] == 'CREATE' and module.params['skip_pending_create']:
                    continue
                if record['PendingChange'] == 'DELETE' and module.params['skip_pending_delete']:
                    continue
            records.append(record)
    if DEFAULTS['as_dict']:
        user_records = {}
        for record in records:
            user_records[record['Username']] = record
        return camel_dict_to_snake_dict(user_records, ignore_list=['Tags'])
    return camel_dict_to_snake_dict(records, ignore_list=['Tags'])
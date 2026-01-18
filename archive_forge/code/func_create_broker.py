from time import sleep
from time import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def create_broker(conn, module):
    kwargs = _fill_kwargs(module)
    wait = module.params.get('wait')
    if 'EngineVersion' in kwargs and kwargs['EngineVersion'] == 'latest':
        kwargs['EngineVersion'] = get_latest_engine_version(conn, module, kwargs['EngineType'])
    if kwargs['AuthenticationStrategy'] == 'LDAP':
        module.fail_json(msg="'AuthenticationStrategy=LDAP' not supported, yet")
    if 'Users' not in kwargs:
        kwargs['Users'] = [{'Username': 'admin', 'Password': 'adminPassword', 'ConsoleAccess': True, 'Groups': []}]
    if 'EncryptionOptions' in kwargs and 'UseAwsOwnedKey' in kwargs['EncryptionOptions']:
        kwargs['EncryptionOptions']['UseAwsOwnedKey'] = False
    if 'SecurityGroups' not in kwargs or len(kwargs['SecurityGroups']) == 0:
        module.fail_json(msg='At least one security group must be specified on broker creation')
    changed = True
    result = conn.create_broker(**kwargs)
    if wait:
        wait_for_status(conn, module)
    return {'broker': camel_dict_to_snake_dict(result, ignore_list=['Tags']), 'changed': changed}
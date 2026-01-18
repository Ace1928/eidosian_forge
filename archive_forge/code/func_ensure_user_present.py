import secrets
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def ensure_user_present(conn, module):
    user = get_matching_user(conn, module, module.params['broker_id'], module.params['username'])
    changed = False
    if user is None:
        if not module.check_mode:
            _response = _create_user(conn, module)
        changed = True
    else:
        kwargs = {}
        if 'groups' in module.params and module.params['groups'] is not None:
            if _group_change_required(user, module.params['groups']):
                kwargs['Groups'] = module.params['groups']
        if 'console_access' in module.params and module.params['console_access'] is not None:
            if _console_access_change_required(user, module.params['console_access']):
                kwargs['ConsoleAccess'] = module.params['console_access']
        if 'password' in module.params and module.params['password']:
            if 'allow_pw_update' in module.params and module.params['allow_pw_update']:
                kwargs['Password'] = module.params['password']
        if len(kwargs) == 0:
            changed = False
        else:
            if not module.check_mode:
                kwargs['BrokerId'] = module.params['broker_id']
                kwargs['Username'] = module.params['username']
                response = _update_user(conn, module, kwargs)
            changed = True
    user = get_matching_user(conn, module, module.params['broker_id'], module.params['username'])
    return {'changed': changed, 'user': camel_dict_to_snake_dict(user, ignore_list=['Tags'])}
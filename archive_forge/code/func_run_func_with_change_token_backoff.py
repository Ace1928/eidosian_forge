from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from .retries import AWSRetry
from .waiters import get_waiter
@AWSRetry.jittered_backoff(backoff=2, catch_extra_error_codes=['WAFStaleDataException'])
def run_func_with_change_token_backoff(client, module, params, func, wait=False):
    params['ChangeToken'] = get_change_token(client, module)
    result = func(**params)
    if wait:
        get_waiter(client, 'change_token_in_sync').wait(ChangeToken=result['ChangeToken'])
    return result
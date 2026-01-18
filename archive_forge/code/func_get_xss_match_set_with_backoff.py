from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from .retries import AWSRetry
from .waiters import get_waiter
@AWSRetry.jittered_backoff(delay=5)
def get_xss_match_set_with_backoff(client, xss_match_set_id):
    return client.get_xss_match_set(XssMatchSetId=xss_match_set_id)['XssMatchSet']
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from .retries import AWSRetry
from .waiters import get_waiter
@AWSRetry.jittered_backoff(delay=5)
def get_byte_match_set_with_backoff(client, byte_match_set_id):
    return client.get_byte_match_set(ByteMatchSetId=byte_match_set_id)['ByteMatchSet']
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from .retries import AWSRetry
from .waiters import get_waiter
@AWSRetry.jittered_backoff(delay=5)
def get_size_constraint_set_with_backoff(client, size_constraint_set_id):
    return client.get_size_constraint_set(SizeConstraintSetId=size_constraint_set_id)['SizeConstraintSet']
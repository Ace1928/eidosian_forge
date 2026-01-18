from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from .retries import AWSRetry
from .waiters import get_waiter
@AWSRetry.jittered_backoff(delay=5)
def list_regional_web_acls_with_backoff(client):
    resp = client.list_web_acls()
    acls = []
    while resp:
        acls += resp['WebACLs']
        resp = client.list_web_acls(NextMarker=resp['NextMarker']) if 'NextMarker' in resp else None
    return acls
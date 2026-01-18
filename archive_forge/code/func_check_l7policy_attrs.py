from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
def check_l7policy_attrs(attrs):
    msg = None
    if 'action' not in attrs:
        return
    if attrs['action'] == 'REDIRECT_TO_POOL':
        if 'redirect_pool_id' not in attrs:
            msg = 'Missing argument: --redirect-pool'
    elif attrs['action'] == 'REDIRECT_TO_URL':
        if 'redirect_url' not in attrs:
            msg = 'Missing argument: --redirect-url'
    elif attrs['action'] == 'REDIRECT_PREFIX':
        if 'redirect_prefix' not in attrs:
            msg = 'Missing argument: --redirect-prefix'
    if msg is not None:
        raise exceptions.CommandError(msg)
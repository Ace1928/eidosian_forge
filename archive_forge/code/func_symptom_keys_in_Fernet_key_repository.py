import keystone.conf
from keystone.common import fernet_utils as utils
def symptom_keys_in_Fernet_key_repository():
    """Fernet key repository is empty.

    After configuring keystone to use the Fernet token provider, you should use
    `keystone-manage fernet_setup` to initially populate your key repository
    with keys, and periodically rotate your keys with `keystone-manage
    fernet_rotate`.
    """
    fernet_utils = utils.FernetUtils(CONF.fernet_tokens.key_repository, CONF.fernet_tokens.max_active_keys, 'fernet_tokens')
    return 'fernet' in CONF.token.provider and (not fernet_utils.load_keys())
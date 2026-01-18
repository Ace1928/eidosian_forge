import keystone.conf
from keystone.common import fernet_utils as utils
from keystone.credential.providers import fernet as credential_fernet
def symptom_keys_in_credential_fernet_key_repository():
    """Credential key repository is empty.

    After configuring keystone to use the Fernet credential provider, you
    should use `keystone-manage credential_setup` to initially populate your
    key repository with keys, and periodically rotate your keys with
    `keystone-manage credential_rotate`.
    """
    fernet_utils = utils.FernetUtils(CONF.credential.key_repository, credential_fernet.MAX_ACTIVE_KEYS, 'credential')
    return 'fernet' in CONF.credential.provider and (not fernet_utils.load_keys())
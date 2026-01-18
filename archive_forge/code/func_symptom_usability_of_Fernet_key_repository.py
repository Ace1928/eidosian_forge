import keystone.conf
from keystone.common import fernet_utils as utils
def symptom_usability_of_Fernet_key_repository():
    """Fernet key repository is not setup correctly.

    The Fernet key repository is expected to be readable by the user running
    keystone, but not world-readable, because it contains security-sensitive
    secrets.
    """
    fernet_utils = utils.FernetUtils(CONF.fernet_tokens.key_repository, CONF.fernet_tokens.max_active_keys, 'fernet_tokens')
    return 'fernet' in CONF.token.provider and (not fernet_utils.validate_key_repository())
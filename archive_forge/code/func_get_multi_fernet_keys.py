import hashlib
from cryptography import fernet
from oslo_log import log
from keystone.common import fernet_utils
import keystone.conf
from keystone.credential.providers import core
from keystone import exception
from keystone.i18n import _
def get_multi_fernet_keys():
    key_utils = fernet_utils.FernetUtils(CONF.credential.key_repository, MAX_ACTIVE_KEYS, 'credential')
    keys = key_utils.load_keys(use_null_key=True)
    fernet_keys = [fernet.Fernet(key) for key in keys]
    crypto = fernet.MultiFernet(fernet_keys)
    return (crypto, keys)
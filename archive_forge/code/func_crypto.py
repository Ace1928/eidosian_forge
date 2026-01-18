import base64
import datetime
import struct
import uuid
from cryptography import fernet
import msgpack
from oslo_log import log
from oslo_utils import timeutils
from keystone.auth import plugins as auth_plugins
from keystone.common import fernet_utils as utils
from keystone.common import utils as ks_utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
@property
def crypto(self):
    """Return a cryptography instance.

        You can extend this class with a custom crypto @property to provide
        your own receipt encoding / decoding. For example, using a different
        cryptography library (e.g. ``python-keyczar``) or to meet arbitrary
        security requirements.

        This @property just needs to return an object that implements
        ``encrypt(plaintext)`` and ``decrypt(ciphertext)``.

        """
    fernet_utils = utils.FernetUtils(CONF.fernet_receipts.key_repository, CONF.fernet_receipts.max_active_keys, 'fernet_receipts')
    keys = fernet_utils.load_keys()
    if not keys:
        raise exception.KeysNotFound()
    fernet_instances = [fernet.Fernet(key) for key in keys]
    return fernet.MultiFernet(fernet_instances)
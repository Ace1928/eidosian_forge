import base64
import datetime
import hashlib
import os
from unittest import mock
import uuid
from oslo_utils import timeutils
from keystone.common import fernet_utils
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.identity.backends import resource_options as ro
from keystone.receipt.providers import fernet
from keystone.receipt import receipt_formatters
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.token import provider as token_provider
@property
def key_repository_signature(self):
    """Create a "thumbprint" of the current key repository.

        Because key files are renamed, this produces a hash of the contents of
        the key files, ignoring their filenames.

        The resulting signature can be used, for example, to ensure that you
        have a unique set of keys after you perform a key rotation (taking a
        static set of keys, and simply shuffling them, would fail such a test).

        """
    key_utils = fernet_utils.FernetUtils(CONF.fernet_receipts.key_repository, CONF.fernet_receipts.max_active_keys, 'fernet_receipts')
    keys = key_utils.load_keys()
    keys.sort()
    signature = hashlib.sha1()
    for key in keys:
        signature.update(key.encode('utf-8'))
    return signature.hexdigest()
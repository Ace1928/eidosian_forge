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
def test_empty_files(self):
    empty_file = os.path.join(CONF.fernet_receipts.key_repository, '2')
    with open(empty_file, 'w'):
        pass
    key_utils = fernet_utils.FernetUtils(CONF.fernet_receipts.key_repository, CONF.fernet_receipts.max_active_keys, 'fernet_receipts')
    keys = key_utils.load_keys()
    self.assertEqual(2, len(keys))
    self.assertValidFernetKeys(keys)
import base64
import datetime
import hashlib
import os
from unittest import mock
import uuid
import fixtures
from oslo_log import log
from oslo_utils import timeutils
from keystone import auth
from keystone.common import fernet_utils
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.federation import constants as federation_constants
from keystone.models import token_model
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.token import provider
from keystone.token.providers import fernet
from keystone.token import token_formatters
def test_convert_or_decode_text_type(self):
    payload_cls = token_formatters.BasePayload
    expected_hex_uuid = uuid.uuid4().hex
    actual_hex_uuid = payload_cls._convert_or_decode(is_stored_as_bytes=False, value=expected_hex_uuid)
    self.assertEqual(expected_hex_uuid, actual_hex_uuid)
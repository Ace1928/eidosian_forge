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
def test_create_validate_federated_unscoped_token_non_uuid_user_id(self):
    exp_user_id = hashlib.sha256().hexdigest()
    exp_methods = ['password']
    exp_expires_at = utils.isotime(timeutils.utcnow(), subsecond=True)
    exp_audit_ids = [provider.random_urlsafe_str()]
    exp_federated_group_ids = [{'id': uuid.uuid4().hex}]
    exp_idp_id = uuid.uuid4().hex
    exp_protocol_id = uuid.uuid4().hex
    token_formatter = token_formatters.TokenFormatter()
    token = token_formatter.create_token(user_id=exp_user_id, expires_at=exp_expires_at, audit_ids=exp_audit_ids, payload_class=token_formatters.FederatedUnscopedPayload, methods=exp_methods, federated_group_ids=exp_federated_group_ids, identity_provider_id=exp_idp_id, protocol_id=exp_protocol_id)
    user_id, methods, audit_ids, system, domain_id, project_id, trust_id, federated_group_ids, identity_provider_id, protocol_id, access_token_id, app_cred_id, thumbprint, issued_at, expires_at = token_formatter.validate_token(token)
    self.assertEqual(exp_user_id, user_id)
    self.assertTrue(isinstance(user_id, str))
    self.assertEqual(exp_methods, methods)
    self.assertEqual(exp_audit_ids, audit_ids)
    self.assertEqual(exp_federated_group_ids, federated_group_ids)
    self.assertEqual(exp_idp_id, identity_provider_id)
    self.assertEqual(exp_protocol_id, protocol_id)
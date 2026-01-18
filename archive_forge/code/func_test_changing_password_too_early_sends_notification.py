import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
import http.client
from oslo_config import fixture as config_fixture
from oslo_log import log
import oslo_messaging
from pycadf import cadftaxonomy
from pycadf import cadftype
from pycadf import eventfactory
from pycadf import resource as cadfresource
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_changing_password_too_early_sends_notification(self):
    password = uuid.uuid4().hex
    new_password = uuid.uuid4().hex
    next_password = uuid.uuid4().hex
    user_ref = unit.new_user_ref(domain_id=self.domain_id, password=password, password_created_at=datetime.datetime.utcnow())
    user_ref = PROVIDERS.identity_api.create_user(user_ref)
    min_days = CONF.security_compliance.minimum_password_age
    min_age = user_ref['password_created_at'] + datetime.timedelta(days=min_days)
    days_left = (min_age - datetime.datetime.utcnow()).days
    reason_type = exception.PasswordAgeValidationError.message_format % {'min_age_days': min_days, 'days_left': days_left}
    expected_reason = {'reasonCode': '400', 'reasonType': reason_type}
    with self.make_request():
        PROVIDERS.identity_api.change_password(user_id=user_ref['id'], original_password=password, new_password=new_password)
    with self.make_request():
        self.assertRaises(exception.PasswordValidationError, PROVIDERS.identity_api.change_password, user_id=user_ref['id'], original_password=new_password, new_password=next_password)
    self._assert_last_audit(user_ref['id'], UPDATED_OPERATION, 'user', cadftaxonomy.SECURITY_ACCOUNT_USER, reason=expected_reason)
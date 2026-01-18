import copy
from unittest import mock
import uuid
from oslo_config import cfg
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
def test_escaped_sequence_in_domain_config(self):
    """Check that escaped '%(' doesn't get interpreted."""
    mock_log = mock.Mock()
    escaped_option_config = {'ldap': {'url': 'my_url/%%(password)s', 'user_tree_dn': uuid.uuid4().hex, 'password': uuid.uuid4().hex}, 'identity': {'driver': uuid.uuid4().hex}}
    PROVIDERS.domain_config_api.create_config(self.domain['id'], escaped_option_config)
    with mock.patch('keystone.resource.core.LOG', mock_log):
        res = PROVIDERS.domain_config_api.get_config_with_sensitive_info(self.domain['id'])
    self.assertFalse(mock_log.warn.called)
    self.assertEqual('my_url/%(password)s', res['ldap']['url'])
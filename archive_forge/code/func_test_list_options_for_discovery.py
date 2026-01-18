import calendar
from unittest import mock
from barbicanclient import exceptions as barbican_exceptions
from keystoneauth1 import identity
from keystoneauth1 import service_token
from oslo_context import context
from oslo_utils import timeutils
from oslo_utils import uuidutils
from castellan.common import exception
from castellan.common.objects import symmetric_key as sym_key
from castellan.key_manager import barbican_key_manager
from castellan.tests.unit.key_manager import test_key_manager
def test_list_options_for_discovery(self):
    opts = self.key_mgr.list_options_for_discovery()
    expected_sections = ['barbican', 'barbican_service_user']
    self.assertEqual(expected_sections, [section[0] for section in opts])
    barbican_opts = [opt.name for opt in opts[0][1]]
    self.assertIn('barbican_endpoint', barbican_opts)
    barbican_service_user_opts = [opt.name for opt in opts[1][1]]
    self.assertIn('cafile', barbican_service_user_opts)
    self.assertIn('auth_section', barbican_service_user_opts)
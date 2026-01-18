import itertools
import os
from unittest import mock
import uuid
from oslo_config import fixture as config_fixture
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import identity
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
def test_config_for_dot_name_domain(self):
    domain_config_filename = os.path.join(self.tmp_dir, 'keystone.abc.def.com.conf')
    with open(domain_config_filename, 'w'):
        'Write an empty config file.'
    self.addCleanup(os.remove, domain_config_filename)
    with mock.patch.object(identity.DomainConfigs, '_load_config_from_file') as mock_load_config:
        domain_config = identity.DomainConfigs()
        fake_assignment_api = None
        fake_standard_driver = None
        domain_config.setup_domain_drivers(fake_standard_driver, fake_assignment_api)
        mock_load_config.assert_called_once_with(fake_assignment_api, [domain_config_filename], 'abc.def.com')
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
def test_config_for_nonexistent_domain(self):
    """Having a config for a non-existent domain will be ignored.

        There are no assertions in this test because there are no side
        effects. If there is a config file for a domain that does not
        exist it should be ignored.

        """
    domain_id = uuid.uuid4().hex
    domain_config_filename = os.path.join(self.tmp_dir, 'keystone.%s.conf' % domain_id)
    self.addCleanup(lambda: os.remove(domain_config_filename))
    with open(domain_config_filename, 'w'):
        'Write an empty config file.'
    e = exception.DomainNotFound(domain_id=domain_id)
    mock_assignment_api = mock.Mock()
    mock_assignment_api.get_domain_by_name.side_effect = e
    domain_config = identity.DomainConfigs()
    fake_standard_driver = None
    domain_config.setup_domain_drivers(fake_standard_driver, mock_assignment_api)
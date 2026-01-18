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
def test_config_for_multiple_sql_backend(self):
    domains_config = identity.DomainConfigs()
    drivers = []
    files = []
    for idx, is_sql in enumerate((True, False, True)):
        drv = mock.Mock(is_sql=is_sql)
        drivers.append(drv)
        name = 'dummy.{0}'.format(idx)
        files.append(''.join((identity.DOMAIN_CONF_FHEAD, name, identity.DOMAIN_CONF_FTAIL)))

    def walk_fake(*a, **kwa):
        return (('/fake/keystone/domains/config', [], files),)
    generic_driver = mock.Mock(is_sql=False)
    assignment_api = mock.Mock()
    id_factory = itertools.count()
    assignment_api.get_domain_by_name.side_effect = lambda name: {'id': next(id_factory), '_': 'fake_domain'}
    load_driver_mock = mock.Mock(side_effect=drivers)
    with mock.patch.object(os, 'walk', walk_fake):
        with mock.patch.object(identity.cfg, 'ConfigOpts'):
            with mock.patch.object(domains_config, '_load_driver', load_driver_mock):
                self.assertRaises(exception.MultipleSQLDriversInConfig, domains_config.setup_domain_drivers, generic_driver, assignment_api)
                self.assertEqual(3, load_driver_mock.call_count)
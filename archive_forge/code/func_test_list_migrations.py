from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import migrations
def test_list_migrations(self):
    ml = self.cs.migrations.list()
    self.assert_request_id(ml, fakes.FAKE_REQUEST_ID_LIST)
    self.cs.assert_called('GET', '/os-migrations')
    for m in ml:
        self.assertIsInstance(m, migrations.Migration)
        self.assertEqual(m.migration_type, 'live-migration')
        self.assertTrue(hasattr(m, 'uuid'))
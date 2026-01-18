import fixtures
from glance.common import exception
from glance.common import swift_store_utils
from glance.tests.unit import base
def test_multiple_swift_account_enabled(self):
    self.config(swift_store_config_file='glance-swift.conf')
    self.assertTrue(swift_store_utils.is_multiple_swift_store_accounts_enabled())
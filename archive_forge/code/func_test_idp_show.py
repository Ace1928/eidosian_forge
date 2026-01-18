from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional.identity.v3 import common
def test_idp_show(self):
    identity_provider = self._create_dummy_idp(add_clean_up=True)
    raw_output = self.openstack('identity provider show %s' % identity_provider)
    items = self.parse_show(raw_output)
    self.assert_show_fields(items, self.IDENTITY_PROVIDER_FIELDS)
from unittest import mock
from novaclient import api_versions
from novaclient import exceptions as exc
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import versions
@mock.patch.object(versions.VersionManager, '_get', side_effect=exc.Unauthorized('401 RAX'))
def test_get_current_with_rax_workaround(self, get):
    self.cs.callback = []
    self.assertIsNone(self.cs.versions.get_current())
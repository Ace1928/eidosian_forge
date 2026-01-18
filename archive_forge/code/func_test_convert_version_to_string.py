from oslotest import base as test_base
from oslo_utils import versionutils
def test_convert_version_to_string(self):
    self.assertEqual('6.7.0', versionutils.convert_version_to_str(6007000))
    self.assertEqual('4', versionutils.convert_version_to_str(4))
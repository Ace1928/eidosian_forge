from oslotest import base as test_base
from oslo_utils import versionutils
def test_convert_version_to_int(self):
    self.assertEqual(6002000, versionutils.convert_version_to_int('6.2.0'))
    self.assertEqual(6004003, versionutils.convert_version_to_int((6, 4, 3)))
    self.assertEqual(5, versionutils.convert_version_to_int((5,)))
    self.assertRaises(ValueError, versionutils.convert_version_to_int, '5a.6b')
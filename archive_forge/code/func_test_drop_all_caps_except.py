from unittest import mock
from oslotest import base
from oslo_privsep import capabilities
@mock.patch('oslo_privsep.capabilities._capset')
def test_drop_all_caps_except(self, mock_capset):
    mock_capset.return_value = 0
    capabilities.drop_all_caps_except((17, 24, 49), (8, 10, 35, 56), (24, 31, 40))
    self.assertEqual(1, mock_capset.call_count)
    hdr, data = mock_capset.call_args[0]
    self.assertEqual(537333798, hdr.version)
    self.assertEqual(16908288, data[0].effective)
    self.assertEqual(131072, data[1].effective)
    self.assertEqual(1280, data[0].permitted)
    self.assertEqual(16777224, data[1].permitted)
    self.assertEqual(2164260864, data[0].inheritable)
    self.assertEqual(256, data[1].inheritable)
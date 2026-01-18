import json
import os
from unittest import mock
import testtools
from openstack.baremetal import configdrive
def test_genisoimage_fails(self, mock_popen):
    mock_popen.return_value.communicate.return_value = (b'', b'BOOM')
    mock_popen.return_value.returncode = 1
    self.assertRaisesRegex(RuntimeError, 'BOOM', configdrive.pack, '/fake')
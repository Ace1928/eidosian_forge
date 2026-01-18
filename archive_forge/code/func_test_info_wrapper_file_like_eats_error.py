import io
import os
import re
import struct
import subprocess
import tempfile
from unittest import mock
from oslo_utils import units
from glance.common import format_inspector
from glance.tests import utils as test_utils
def test_info_wrapper_file_like_eats_error(self):
    wrapper = self._get_wrapper(b'123456')
    wrapper._format.eat_chunk.side_effect = Exception('fail')
    data = b''
    while True:
        chunk = wrapper.read(3)
        if not chunk:
            break
        data += chunk
    self.assertEqual(b'123456', data)
    wrapper._format.eat_chunk.assert_called_once_with(b'123')
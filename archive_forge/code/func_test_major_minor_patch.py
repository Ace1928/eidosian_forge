import unittest
from numba.tests.support import TestCase
from numba.misc.init_utils import version_info, generate_version_info
def test_major_minor_patch(self):
    expected = version_info(0, 1, 0, (0, 1), (0, 1, 0), '0.1.0', ('0', '1', '0'), None)
    received = generate_version_info('0.1.0')
    self.assertEqual(received, expected)
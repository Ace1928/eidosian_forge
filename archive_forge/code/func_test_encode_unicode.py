import pathlib
import unittest
from pygame import encode_string, encode_file_path
def test_encode_unicode(self):
    u = 'Þe Olde Kompüter Shoppe'
    b = u.encode('utf-8')
    self.assertEqual(encode_string(u, 'utf-8'), b)
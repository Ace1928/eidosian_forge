import unittest
from charset_normalizer.legacy import detect
def test_detect_dict_value(self):
    r = detect('我没有埋怨，磋砣的只是一些时间。'.encode('utf_32'))
    with self.subTest('encoding is equal to utf_32'):
        self.assertEqual(r['encoding'], 'UTF-32')
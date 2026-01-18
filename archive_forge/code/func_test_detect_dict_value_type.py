import unittest
from charset_normalizer.legacy import detect
def test_detect_dict_value_type(self):
    r = detect('我没有埋怨，磋砣的只是一些时间。'.encode('utf_8'))
    with self.subTest('encoding instance of str'):
        self.assertIsInstance(r['encoding'], str)
    with self.subTest('language instance of str'):
        self.assertIsInstance(r['language'], str)
    with self.subTest('confidence instance of float'):
        self.assertIsInstance(r['confidence'], float)
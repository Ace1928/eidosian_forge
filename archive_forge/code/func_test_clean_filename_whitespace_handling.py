import datetime
import unittest
from traits.util.clean_strings import clean_filename, clean_timestamp
def test_clean_filename_whitespace_handling(self):
    with self.assertWarns(DeprecationWarning):
        self.assertEqual(clean_filename(' abc '), 'abc')
    with self.assertWarns(DeprecationWarning):
        self.assertEqual(clean_filename(' \t\tabc    \n'), 'abc')
    with self.assertWarns(DeprecationWarning):
        self.assertEqual(clean_filename('well name'), 'well-name')
    with self.assertWarns(DeprecationWarning):
        self.assertEqual(clean_filename('well \n name'), 'well-name')
    with self.assertWarns(DeprecationWarning):
        self.assertEqual(clean_filename('well - name'), 'well-name')
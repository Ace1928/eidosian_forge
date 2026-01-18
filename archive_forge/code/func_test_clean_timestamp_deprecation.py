import datetime
import unittest
from traits.util.clean_strings import clean_filename, clean_timestamp
def test_clean_timestamp_deprecation(self):
    with self.assertWarns(DeprecationWarning):
        clean_timestamp(datetime.datetime.now())
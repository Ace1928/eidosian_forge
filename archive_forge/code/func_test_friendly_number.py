import datetime
import os
import shutil
import tempfile
import unittest
import tornado.locale
from tornado.escape import utf8, to_unicode
from tornado.util import unicode_type
def test_friendly_number(self):
    locale = tornado.locale.get('en_US')
    self.assertEqual(locale.friendly_number(1000000), '1,000,000')
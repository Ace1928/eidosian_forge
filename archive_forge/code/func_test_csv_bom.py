import datetime
import os
import shutil
import tempfile
import unittest
import tornado.locale
from tornado.escape import utf8, to_unicode
from tornado.util import unicode_type
def test_csv_bom(self):
    with open(os.path.join(os.path.dirname(__file__), 'csv_translations', 'fr_FR.csv'), 'rb') as f:
        char_data = to_unicode(f.read())
    for encoding in ['utf-8-sig', 'utf-16']:
        tmpdir = tempfile.mkdtemp()
        try:
            with open(os.path.join(tmpdir, 'fr_FR.csv'), 'wb') as f:
                f.write(char_data.encode(encoding))
            tornado.locale.load_translations(tmpdir)
            locale = tornado.locale.get('fr_FR')
            self.assertIsInstance(locale, tornado.locale.CSVLocale)
            self.assertEqual(locale.translate('school'), 'école')
        finally:
            shutil.rmtree(tmpdir)
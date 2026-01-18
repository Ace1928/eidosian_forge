import doctest
import os
import shutil
import tempfile
import unittest
from genshi.core import TEXT
from genshi.template.loader import TemplateLoader
from genshi.template.markup import MarkupTemplate
def test_load_with_default_encoding(self):
    f = open(os.path.join(self.dirname, 'tmpl.html'), 'wb')
    try:
        f.write(u'<div>ö</div>'.encode('iso-8859-1'))
    finally:
        f.close()
    loader = TemplateLoader([self.dirname], default_encoding='iso-8859-1')
    loader.load('tmpl.html')
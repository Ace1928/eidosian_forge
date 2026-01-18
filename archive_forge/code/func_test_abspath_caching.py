import doctest
import os
import shutil
import tempfile
import unittest
from genshi.core import TEXT
from genshi.template.loader import TemplateLoader
from genshi.template.markup import MarkupTemplate
def test_abspath_caching(self):
    abspath = os.path.join(self.dirname, 'abs')
    os.mkdir(abspath)
    file1 = open(os.path.join(abspath, 'tmpl1.html'), 'w')
    try:
        file1.write('<html xmlns:xi="http://www.w3.org/2001/XInclude">\n              <xi:include href="tmpl2.html" />\n            </html>')
    finally:
        file1.close()
    file2 = open(os.path.join(abspath, 'tmpl2.html'), 'w')
    try:
        file2.write('<div>Included from abspath.</div>')
    finally:
        file2.close()
    searchpath = os.path.join(self.dirname, 'searchpath')
    os.mkdir(searchpath)
    file3 = open(os.path.join(searchpath, 'tmpl2.html'), 'w')
    try:
        file3.write('<div>Included from searchpath.</div>')
    finally:
        file3.close()
    loader = TemplateLoader(searchpath)
    tmpl1 = loader.load(os.path.join(abspath, 'tmpl1.html'))
    self.assertEqual('<html>\n              <div>Included from searchpath.</div>\n            </html>', tmpl1.generate().render(encoding=None))
    assert 'tmpl2.html' in loader._cache
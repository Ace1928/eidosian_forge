import doctest
import os
import pickle
import shutil
import sys
import tempfile
import unittest
import six
from genshi.compat import BytesIO, StringIO
from genshi.core import Markup
from genshi.filters.i18n import Translator
from genshi.input import XML
from genshi.template.base import BadDirectiveError, TemplateSyntaxError
from genshi.template.loader import TemplateLoader, TemplateNotFound
from genshi.template.markup import MarkupTemplate
def test_nested_include_matches(self):
    dirname = tempfile.mkdtemp(suffix='genshi_test')
    try:
        file1 = open(os.path.join(dirname, 'tmpl1.html'), 'w')
        try:
            file1.write('<html xmlns:py="http://genshi.edgewall.org/" py:strip="">\n   <div class="target">Some content.</div>\n</html>')
        finally:
            file1.close()
        file2 = open(os.path.join(dirname, 'tmpl2.html'), 'w')
        try:
            file2.write('<html xmlns:py="http://genshi.edgewall.org/"\n    xmlns:xi="http://www.w3.org/2001/XInclude">\n  <body>\n    <h1>Some full html document that includes file1.html</h1>\n    <xi:include href="tmpl1.html" />\n  </body>\n</html>')
        finally:
            file2.close()
        file3 = open(os.path.join(dirname, 'tmpl3.html'), 'w')
        try:
            file3.write('<html xmlns:py="http://genshi.edgewall.org/"\n    xmlns:xi="http://www.w3.org/2001/XInclude" py:strip="">\n  <div py:match="div[@class=\'target\']" py:attrs="select(\'@*\')">\n    Some added stuff.\n    ${select(\'*|text()\')}\n  </div>\n  <xi:include href="tmpl2.html" />\n</html>\n')
        finally:
            file3.close()
        loader = TemplateLoader([dirname])
        tmpl = loader.load('tmpl3.html')
        self.assertEqual('\n  <html>\n  <body>\n    <h1>Some full html document that includes file1.html</h1>\n   <div class="target">\n    Some added stuff.\n    Some content.\n  </div>\n  </body>\n</html>\n', tmpl.generate().render(encoding=None))
    finally:
        shutil.rmtree(dirname)
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
def test_include_in_fallback(self):
    dirname = tempfile.mkdtemp(suffix='genshi_test')
    try:
        file1 = open(os.path.join(dirname, 'tmpl1.html'), 'w')
        try:
            file1.write('<div>Included</div>')
        finally:
            file1.close()
        file2 = open(os.path.join(dirname, 'tmpl3.html'), 'w')
        try:
            file2.write('<html xmlns:xi="http://www.w3.org/2001/XInclude">\n                  <xi:include href="tmpl2.html">\n                    <xi:fallback>\n                      <xi:include href="tmpl1.html">\n                        <xi:fallback>Missing</xi:fallback>\n                      </xi:include>\n                    </xi:fallback>\n                  </xi:include>\n                </html>')
        finally:
            file2.close()
        loader = TemplateLoader([dirname])
        tmpl = loader.load('tmpl3.html')
        self.assertEqual('<html>\n                      <div>Included</div>\n                </html>', tmpl.generate().render(encoding=None))
    finally:
        shutil.rmtree(dirname)
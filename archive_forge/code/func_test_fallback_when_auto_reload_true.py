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
def test_fallback_when_auto_reload_true(self):
    dirname = tempfile.mkdtemp(suffix='genshi_test')
    try:
        file2 = open(os.path.join(dirname, 'tmpl2.html'), 'w')
        try:
            file2.write('<html xmlns:xi="http://www.w3.org/2001/XInclude">\n                  <xi:include href="tmpl1.html"><xi:fallback>\n                    Missing</xi:fallback></xi:include>\n                </html>')
        finally:
            file2.close()
        loader = TemplateLoader([dirname], auto_reload=True)
        tmpl = loader.load('tmpl2.html')
        self.assertEqual('<html>\n                    Missing\n                </html>', tmpl.generate().render(encoding=None))
    finally:
        shutil.rmtree(dirname)
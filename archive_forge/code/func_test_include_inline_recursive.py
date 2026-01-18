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
def test_include_inline_recursive(self):
    dirname = tempfile.mkdtemp(suffix='genshi_test')
    try:
        file1 = open(os.path.join(dirname, 'tmpl1.html'), 'w')
        try:
            file1.write('<div xmlns:xi="http://www.w3.org/2001/XInclude"                xmlns:py="http://genshi.edgewall.org/">$depth<py:with vars="depth = depth + 1"><xi:include href="tmpl1.html"            py:if="depth &lt; 3"/></py:with></div>')
        finally:
            file1.close()
        loader = TemplateLoader([dirname], auto_reload=False)
        tmpl = loader.load(os.path.join(dirname, 'tmpl1.html'))
        self.assertEqual('<div>0<div>1<div>2</div></div></div>', tmpl.generate(depth=0).render(encoding=None))
    finally:
        shutil.rmtree(dirname)
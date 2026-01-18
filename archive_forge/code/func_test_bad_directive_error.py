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
def test_bad_directive_error(self):
    xml = '<p xmlns:py="http://genshi.edgewall.org/" py:do="nothing" />'
    try:
        tmpl = MarkupTemplate(xml, filename='test.html')
    except BadDirectiveError as e:
        self.assertEqual('test.html', e.filename)
        self.assertEqual(1, e.lineno)
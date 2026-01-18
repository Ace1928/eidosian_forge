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
def test_match_without_select(self):
    xml = '<html xmlns:py="http://genshi.edgewall.org/">\n          <py:match path="body" buffer="false">\n            <body>\n              This replaces the other text.\n            </body>\n          </py:match>\n          <body>\n            This gets replaced.\n          </body>\n        </html>'
    tmpl = MarkupTemplate(xml, filename='test.html')
    self.assertEqual('<html>\n            <body>\n              This replaces the other text.\n            </body>\n        </html>', tmpl.generate().render(encoding=None))
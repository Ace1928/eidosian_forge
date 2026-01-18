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
def test_exec_in_match(self):
    xml = '<html xmlns:py="http://genshi.edgewall.org/">\n          <py:match path="body/p">\n            <?python title="wakka wakka wakka" ?>\n            ${title}\n          </py:match>\n          <body><p>moot text</p></body>\n        </html>'
    tmpl = MarkupTemplate(xml, filename='test.html', allow_exec=True)
    self.assertEqual('<html>\n          <body>\n            wakka wakka wakka\n          </body>\n        </html>', tmpl.generate().render(encoding=None))
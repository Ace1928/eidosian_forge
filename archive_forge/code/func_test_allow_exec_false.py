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
def test_allow_exec_false(self):
    xml = '<?python\n          title = "A Genshi Template"\n          ?>\n          <html xmlns:py="http://genshi.edgewall.org/">\n            <head>\n              <title py:content="title">This is replaced.</title>\n            </head>\n        </html>'
    try:
        tmpl = MarkupTemplate(xml, filename='test.html', allow_exec=False)
        self.fail('Expected SyntaxError')
    except TemplateSyntaxError as e:
        pass
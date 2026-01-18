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
def test_latin1_encoded_with_xmldecl(self):
    tmpl = MarkupTemplate(u'<?xml version="1.0" encoding="iso-8859-1" ?>\n        <div xmlns:py="http://genshi.edgewall.org/">\n          ö\n        </div>'.encode('iso-8859-1'), encoding='iso-8859-1')
    self.assertEqual(u'<?xml version="1.0" encoding="iso-8859-1"?>\n<div>\n          ö\n        </div>', six.text_type(tmpl.generate()))
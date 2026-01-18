from datetime import datetime
from gettext import NullTranslations
import unittest
import six
from genshi.core import Attrs
from genshi.template import MarkupTemplate, Context
from genshi.filters.i18n import Translator, extract
from genshi.input import HTML
from genshi.compat import IS_PYTHON2, StringIO
from genshi.tests.test_utils import doctest_suite
def test_extraction_without_text(self):
    buf = StringIO('<html xmlns:py="http://genshi.edgewall.org/">\n          <p title="Bar">Foo</p>\n          ${ngettext("Singular", "Plural", num)}\n        </html>')
    results = list(extract(buf, ['_', 'ngettext'], [], {'extract_text': 'no'}))
    self.assertEqual([(3, 'ngettext', ('Singular', 'Plural', None), [])], results)
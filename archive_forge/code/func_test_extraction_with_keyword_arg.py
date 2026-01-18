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
def test_extraction_with_keyword_arg(self):
    buf = StringIO('<html xmlns:py="http://genshi.edgewall.org/">\n          ${gettext(\'Foobar\', foo=\'bar\')}\n        </html>')
    results = list(extract(buf, ['gettext'], [], {}))
    self.assertEqual([(2, 'gettext', 'Foobar', [])], results)
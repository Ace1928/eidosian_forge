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
def test_extract_msgcontext(self):
    buf = StringIO('<html xmlns:py="http://genshi.edgewall.org/"\n                                xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:ctxt="foo">Foo, bar.</p>\n          <p>Foo, bar.</p>\n        </html>')
    results = list(extract(buf, ['_'], [], {}))
    self.assertEqual((3, 'pgettext', ('foo', 'Foo, bar.'), []), results[0])
    self.assertEqual((4, None, 'Foo, bar.', []), results[1])
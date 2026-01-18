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
def test_extraction_with_nonstring_arg(self):
    buf = StringIO('<html xmlns:py="http://genshi.edgewall.org/">\n          ${dgettext(curdomain, \'Foobar\')}\n        </html>')
    results = list(extract(buf, ['dgettext'], [], {}))
    self.assertEqual([(2, 'dgettext', (None, 'Foobar'), [])], results)
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
def test_markup_template_extraction(self):
    buf = StringIO('<html xmlns:py="http://genshi.edgewall.org/">\n          <head>\n            <title>Example</title>\n          </head>\n          <body>\n            <h1>Example</h1>\n            <p>${_("Hello, %(name)s") % dict(name=username)}</p>\n            <p>${ngettext("You have %d item", "You have %d items", num)}</p>\n          </body>\n        </html>')
    results = list(extract(buf, ['_', 'ngettext'], [], {}))
    self.assertEqual([(3, None, 'Example', []), (6, None, 'Example', []), (7, '_', 'Hello, %(name)s', []), (8, 'ngettext', ('You have %d item', 'You have %d items', None), [])], results)
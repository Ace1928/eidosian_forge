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
def test_extract_i18n_msg_with_dynamic_attributes(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="" title="${_(\'A helpful paragraph\')}">\n            Please see <a href="help.html" title="${_(\'Click for help\')}">Help</a>\n          </p>\n        </html>')
    translator = Translator()
    translator.setup(tmpl)
    messages = list(translator.extract(tmpl.stream))
    self.assertEqual(3, len(messages))
    self.assertEqual('A helpful paragraph', messages[0][2])
    self.assertEqual(3, messages[0][0])
    self.assertEqual('Click for help', messages[1][2])
    self.assertEqual(4, messages[1][0])
    self.assertEqual('Please see [1:Help]', messages[2][2])
    self.assertEqual(3, messages[2][0])
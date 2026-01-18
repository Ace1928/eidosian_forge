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
def test_extract_i18n_choose_as_element_with_attributes(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <i18n:choose numeral="num" params="num">\n            <p i18n:singular="" title="Things">\n              There is <a href="$link" title="View thing">${num} thing</a>.\n            </p>\n            <p i18n:plural="" title="Things">\n              There are <a href="$link" title="View things">${num} things</a>.\n            </p>\n          </i18n:choose>\n        </html>')
    translator = Translator()
    translator.setup(tmpl)
    messages = list(translator.extract(tmpl.stream))
    self.assertEqual(5, len(messages))
    self.assertEqual((4, None, 'Things', []), messages[0])
    self.assertEqual((5, None, 'View thing', []), messages[1])
    self.assertEqual((7, None, 'Things', []), messages[2])
    self.assertEqual((8, None, 'View things', []), messages[3])
    self.assertEqual((3, 'ngettext', ('There is [1:%(num)s thing].', 'There are [1:%(num)s things].'), []), messages[4])
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
def test_extract_i18n_choose_as_attribute_and_py_strip(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <div i18n:choose="one" py:strip="">\n            <p i18n:singular="" py:strip="">FooBar</p>\n            <p i18n:plural="" py:strip="">FooBars</p>\n          </div>\n        </html>')
    translator = Translator()
    tmpl.add_directives(Translator.NAMESPACE, translator)
    messages = list(translator.extract(tmpl.stream))
    self.assertEqual(1, len(messages))
    self.assertEqual((3, 'ngettext', ('FooBar', 'FooBars'), []), messages[0])
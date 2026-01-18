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
def test_extract_i18n_choose_as_directive_with_params_and_comment(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n        <i18n:choose numeral="two" params="fname, lname" i18n:comment="As in Foo Bar">\n          <p i18n:singular="">Foo ${fname} ${lname}</p>\n          <p i18n:plural="">Foos ${fname} ${lname}</p>\n        </i18n:choose>\n        </html>')
    translator = Translator()
    tmpl.add_directives(Translator.NAMESPACE, translator)
    messages = list(translator.extract(tmpl.stream))
    self.assertEqual(1, len(messages))
    self.assertEqual((3, 'ngettext', ('Foo %(fname)s %(lname)s', 'Foos %(fname)s %(lname)s'), ['As in Foo Bar']), messages[0])
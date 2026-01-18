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
def test_i18n_msg_ticket_300_extract(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <i18n:msg params="date, author">\n            Changed ${ \'10/12/2008\' } ago by ${ \'me, the author\' }\n          </i18n:msg>\n        </html>')
    translator = Translator()
    tmpl.add_directives(Translator.NAMESPACE, translator)
    messages = list(translator.extract(tmpl.stream))
    self.assertEqual(1, len(messages))
    self.assertEqual((3, None, 'Changed %(date)s ago by %(author)s', []), messages[0])
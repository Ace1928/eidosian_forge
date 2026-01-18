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
def test_i18n_msg_ticket_300_translate(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <i18n:msg params="date, author">\n            Changed ${ date } ago by ${ author }\n          </i18n:msg>\n        </html>')
    translations = DummyTranslations({'Changed %(date)s ago by %(author)s': u'Modificado à %(date)s por %(author)s'})
    translator = Translator(translations)
    translator.setup(tmpl)
    self.assertEqual(u'<html>\n          Modificado à um dia por Pedro\n        </html>'.encode('utf-8'), tmpl.generate(date='um dia', author='Pedro').render(encoding='utf-8'))
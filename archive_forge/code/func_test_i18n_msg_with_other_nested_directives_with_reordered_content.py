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
def test_i18n_msg_with_other_nested_directives_with_reordered_content(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p py:if="not editable" class="hint" i18n:msg="">\n            <strong>Note:</strong> This repository is defined in\n            <code><a href="${ \'href.wiki(TracIni)\' }">trac.ini</a></code>\n            and cannot be edited on this page.\n          </p>\n        </html>')
    translations = DummyTranslations({'[1:Note:] This repository is defined in\n            [2:[3:trac.ini]]\n            and cannot be edited on this page.': u'[1:Nota:] Este repositório está definido em \n           [2:[3:trac.ini]]\n            e não pode ser editado nesta página.'})
    translator = Translator(translations)
    translator.setup(tmpl)
    messages = list(translator.extract(tmpl.stream))
    self.assertEqual(1, len(messages))
    self.assertEqual('[1:Note:] This repository is defined in\n            [2:[3:trac.ini]]\n            and cannot be edited on this page.', messages[0][2])
    self.assertEqual(u'<html>\n          <p class="hint"><strong>Nota:</strong> Este repositório está definido em\n           <code><a href="href.wiki(TracIni)">trac.ini</a></code>\n            e não pode ser editado nesta página.</p>\n        </html>'.encode('utf-8'), tmpl.generate(editable=False).render(encoding='utf-8'))
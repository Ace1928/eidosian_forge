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
def test_extraction_inside_ignored_tags_with_directives(self):
    buf = StringIO('<html xmlns:py="http://genshi.edgewall.org/">\n          <script type="text/javascript">\n            <py:if test="foobar">\n              alert("This shouldn\'t be extracted");\n            </py:if>\n          </script>\n        </html>')
    self.assertEqual([], list(extract(buf, ['_'], [], {})))
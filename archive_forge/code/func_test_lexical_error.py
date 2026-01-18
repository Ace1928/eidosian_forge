import io
import sys
import yaql
from yaql.language import exceptions
from yaql.language import factory
from yaql.language import specs
from yaql.language import yaqltypes
from yaql import tests
def test_lexical_error(self):
    self.assertRaises(exceptions.YaqlLexicalException, self.eval, '1 ? 2')
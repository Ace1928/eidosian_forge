import io
import sys
import yaql
from yaql.language import exceptions
from yaql.language import factory
from yaql.language import specs
from yaql.language import yaqltypes
from yaql import tests
def test_no_trailing_commas(self):
    self.assertRaises(exceptions.YaqlGrammarException, self.eval, 'func(1,,)')
    self.assertRaises(exceptions.YaqlGrammarException, self.eval, 'func(,1,)')
    self.assertRaises(exceptions.YaqlGrammarException, self.eval, 'func(,,)')
    self.assertRaises(exceptions.YaqlGrammarException, self.eval, 'func(,)')
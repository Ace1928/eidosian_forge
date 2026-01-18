import io
import sys
import yaql
from yaql.language import exceptions
from yaql.language import factory
from yaql.language import specs
from yaql.language import yaqltypes
from yaql import tests
def test_no_varargs_after_kwargs(self):
    self.assertRaises(exceptions.YaqlGrammarException, self.eval, 'func(x=>y, t)')
    self.assertRaises(exceptions.YaqlGrammarException, self.eval, 'func(x=>y, ,t)')
    self.assertRaises(exceptions.YaqlGrammarException, self.eval, 'func(a, x=>y, ,t)')
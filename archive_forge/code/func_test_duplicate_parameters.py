import io
import sys
import yaql
from yaql.language import exceptions
from yaql.language import factory
from yaql.language import specs
from yaql.language import yaqltypes
from yaql import tests
def test_duplicate_parameters(self):

    def raises():

        @specs.parameter('p')
        @specs.parameter('p')
        def f(p):
            return p
    self.assertRaises(exceptions.DuplicateParameterDecoratorException, raises)
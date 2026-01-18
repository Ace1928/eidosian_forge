import io
import sys
import yaql
from yaql.language import exceptions
from yaql.language import factory
from yaql.language import specs
from yaql.language import yaqltypes
from yaql import tests
def test_no_matching_function(self):
    self.assertRaises(exceptions.NoMatchingFunctionException, self.eval, 'len(1, 2, 3)')
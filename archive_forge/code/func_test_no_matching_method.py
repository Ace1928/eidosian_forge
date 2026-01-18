import io
import sys
import yaql
from yaql.language import exceptions
from yaql.language import factory
from yaql.language import specs
from yaql.language import yaqltypes
from yaql import tests
def test_no_matching_method(self):
    self.assertRaises(exceptions.NoMatchingMethodException, self.eval, '[1, 2].select(1, 2, 3)')
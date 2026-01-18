import io
import sys
import yaql
from yaql.language import exceptions
from yaql.language import factory
from yaql.language import specs
from yaql.language import yaqltypes
from yaql import tests
@specs.parameter('string', yaqltypes.String())
@specs.inject('base', yaqltypes.Super())
def len2(string, base):
    return 2 * base(string)
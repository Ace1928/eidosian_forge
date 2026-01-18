import sys
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
from yaql.language import yaqltypes
def raise_ambiguous():
    if receiver is utils.NO_VALUE:
        raise exceptions.AmbiguousFunctionException(name)
    else:
        raise exceptions.AmbiguousMethodException(name, receiver)
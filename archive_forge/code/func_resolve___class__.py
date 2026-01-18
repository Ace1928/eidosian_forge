from .. import types, utils, errors
import operator
from .templates import (AttributeTemplate, ConcreteTemplate, AbstractTemplate,
from .builtins import normalize_1d_index
def resolve___class__(self, tup):
    return types.NamedTupleClass(tup.instance_class)
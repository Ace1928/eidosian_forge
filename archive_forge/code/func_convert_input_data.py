import collections
import re
import sys
from yaql.language import exceptions
from yaql.language import lexer
def convert_input_data(obj, rec=None):
    if rec is None:
        rec = convert_input_data
    if isinstance(obj, str):
        return obj if isinstance(obj, str) else str(obj)
    elif isinstance(obj, SequenceType):
        return tuple((rec(t, rec) for t in obj))
    elif isinstance(obj, MappingType):
        return FrozenDict(((rec(key, rec), rec(value, rec)) for key, value in obj.items()))
    elif isinstance(obj, MutableSetType):
        return frozenset((rec(t, rec) for t in obj))
    elif isinstance(obj, IterableType):
        return map(lambda v: rec(v, rec), obj)
    else:
        return obj
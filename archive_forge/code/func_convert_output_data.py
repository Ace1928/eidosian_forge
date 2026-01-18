import collections
import re
import sys
from yaql.language import exceptions
from yaql.language import lexer
def convert_output_data(obj, limit_func, engine, rec=None):
    if rec is None:
        rec = convert_output_data
    if isinstance(obj, collections.abc.Mapping):
        result = {}
        for key, value in limit_func(obj.items()):
            result[rec(key, limit_func, engine, rec)] = rec(value, limit_func, engine, rec)
        return result
    elif isinstance(obj, SetType):
        set_type = list if convert_sets_to_lists(engine) else set
        return set_type((rec(t, limit_func, engine, rec) for t in limit_func(obj)))
    elif isinstance(obj, (tuple, list)):
        seq_type = list if convert_tuples_to_lists(engine) else type(obj)
        return seq_type((rec(t, limit_func, engine, rec) for t in limit_func(obj)))
    elif is_iterable(obj):
        return list((rec(t, limit_func, engine, rec) for t in limit_func(obj)))
    else:
        return obj
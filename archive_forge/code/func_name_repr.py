import re
def name_repr(x, unknown_handler=str):
    if not isinstance(x, str):
        return _repr_map.get(x.__class__, unknown_handler)(x)
    else:
        x = repr(x)
        if x[1] == '|':
            return x
        unquoted = x[1:-1]
        if re_special_char.search(unquoted):
            return x
        if re_number.fullmatch(unquoted):
            return x
        return unquoted
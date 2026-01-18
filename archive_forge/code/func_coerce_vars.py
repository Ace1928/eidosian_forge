import html
from urllib.parse import parse_qsl, quote, unquote, urlencode
from paste import request
def coerce_vars(self, vars):
    global variabledecode
    need_variable_encode = False
    for key, value in vars.items():
        if isinstance(value, dict):
            need_variable_encode = True
        if key.endswith('_'):
            vars[key[:-1]] = vars[key]
            del vars[key]
    if need_variable_encode:
        if variabledecode is None:
            from formencode import variabledecode
        vars = variabledecode.variable_encode(vars)
    return vars
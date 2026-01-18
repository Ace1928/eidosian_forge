import re
from sympy.concrete.products import product
from sympy.concrete.summations import Sum
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import (cos, sin)
def parse_maxima(str, globals=None, name_dict={}):
    str = str.strip()
    str = str.rstrip('; ')
    for k, v in sub_dict.items():
        str = v.sub(k, str)
    assign_var = None
    var_match = var_name.search(str)
    if var_match:
        assign_var = var_match.group(1)
        str = str[var_match.end():].strip()
    dct = MaximaHelpers.__dict__.copy()
    dct.update(name_dict)
    obj = sympify(str, locals=dct)
    if assign_var and globals:
        globals[assign_var] = obj
    return obj
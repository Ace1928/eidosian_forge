from statsmodels.compat.python import lmap, lrange, lzip
import copy
from itertools import zip_longest
import time
import numpy as np
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import (
from .summary2 import _model_types
def summary_top(results, title=None, gleft=None, gright=None, yname=None, xname=None):
    """generate top table(s)


    TODO: this still uses predefined model_methods
    ? allow gleft, gright to be 1 element tuples instead of filling with None?

    """
    gen_left, gen_right = (gleft, gright)
    time_now = time.localtime()
    time_of_day = [time.strftime('%H:%M:%S', time_now)]
    date = time.strftime('%a, %d %b %Y', time_now)
    yname, xname = _getnames(results, yname=yname, xname=xname)
    default_items = dict([('Dependent Variable:', lambda: [yname]), ('Dep. Variable:', lambda: [yname]), ('Model:', lambda: [results.model.__class__.__name__]), ('Date:', lambda: [date]), ('Time:', lambda: time_of_day), ('Number of Obs:', lambda: [results.nobs]), ('No. Observations:', lambda: [d_or_f(results.nobs)]), ('Df Model:', lambda: [d_or_f(results.df_model)]), ('Df Residuals:', lambda: [d_or_f(results.df_resid)]), ('Log-Likelihood:', lambda: ['%#8.5g' % results.llf])])
    if title is None:
        title = results.model.__class__.__name__ + 'Regression Results'
    if gen_left is None:
        gen_left = [('Dep. Variable:', None), ('Model type:', None), ('Date:', None), ('No. Observations:', None), ('Df model:', None), ('Df resid:', None)]
        try:
            llf = results.llf
            gen_left.append(('Log-Likelihood', None))
        except:
            pass
        gen_right = []
    gen_title = title
    gen_header = None
    gen_left_ = []
    for item, value in gen_left:
        if value is None:
            value = default_items[item]()
        gen_left_.append((item, value))
    gen_left = gen_left_
    if gen_right:
        gen_right_ = []
        for item, value in gen_right:
            if value is None:
                value = default_items[item]()
            gen_right_.append((item, value))
        gen_right = gen_right_
    missing_values = [k for k, v in gen_left + gen_right if v is None]
    assert missing_values == [], missing_values
    if gen_right:
        if len(gen_right) < len(gen_left):
            gen_right += [(' ', ' ')] * (len(gen_left) - len(gen_right))
        elif len(gen_right) > len(gen_left):
            gen_left += [(' ', ' ')] * (len(gen_right) - len(gen_left))
        gen_right = [('%-21s' % ('  ' + k), v) for k, v in gen_right]
        gen_stubs_right, gen_data_right = zip_longest(*gen_right)
        gen_table_right = SimpleTable(gen_data_right, gen_header, gen_stubs_right, title=gen_title, txt_fmt=fmt_2cols)
    else:
        gen_table_right = []
    gen_stubs_left, gen_data_left = zip_longest(*gen_left)
    gen_table_left = SimpleTable(gen_data_left, gen_header, gen_stubs_left, title=gen_title, txt_fmt=fmt_2cols)
    gen_table_left.extend_right(gen_table_right)
    general_table = gen_table_left
    return general_table
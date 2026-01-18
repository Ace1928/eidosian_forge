import csv
import io
import math
from tensorboard.plugins.hparams import error
def latex_format(value):
    if value is None:
        return '-'
    elif isinstance(value, int):
        return '$%d$' % value
    elif isinstance(value, float):
        if math.isnan(value):
            return '$\\mathrm{NaN}$'
        if value in (float('inf'), float('-inf')):
            return '$%s\\infty$' % ('-' if value < 0 else '+')
        scientific = '%.3g' % value
        if 'e' in scientific:
            coefficient, exponent = scientific.split('e')
            return '$%s\\cdot 10^{%d}$' % (coefficient, int(exponent))
        return '$%s$' % scientific
    return value.replace('_', '\\_').replace('%', '\\%')
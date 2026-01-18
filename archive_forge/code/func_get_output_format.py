from statsmodels.compat.python import lmap, lrange
from itertools import cycle, zip_longest
import csv
def get_output_format(output_format):
    if output_format not in ('html', 'txt', 'latex', 'csv'):
        try:
            output_format = output_format_translations[output_format]
        except KeyError:
            raise ValueError('unknown output format %s' % output_format)
    return output_format
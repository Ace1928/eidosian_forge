import textwrap
from pprint import PrettyPrinter
from _plotly_utils.utils import *
from _plotly_utils.data_utils import *
@staticmethod
def is_wrappable(v):
    numpy = get_module('numpy')
    if isinstance(v, (list, tuple)) and len(v) > 0 and (not isinstance(v[0], dict)):
        return True
    elif numpy and isinstance(v, numpy.ndarray):
        return True
    elif isinstance(v, str):
        return True
    else:
        return False
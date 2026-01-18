import re
from ... import ndarray, symbol
from .. import HybridBlock, tensor_types
from . import rnn_cell
from ...util import is_np_array
def convert_key(m, bidirectional):
    d, l, g, t = [m.group(i) for i in range(1, 5)]
    if bidirectional:
        return '_unfused.{}.{}_cell.{}_{}'.format(l, d, g, t)
    else:
        return '_unfused.{}.{}_{}'.format(l, g, t)
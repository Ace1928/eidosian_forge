from functools import partial
import pickle
import numpy as np
import pandas as pd
from pandas.core.internals import create_block_manager_from_blocks, make_block
from . import numpy as pnp
from .core import Interface
from .encode import Encode
from .utils import extend, framesplit, frame
def index_to_header_bytes(ind):
    if isinstance(ind, (pd.DatetimeIndex, pd.MultiIndex, pd.RangeIndex)):
        return (None, dumps(ind))
    if isinstance(ind, pd.CategoricalIndex):
        cat = (ind.ordered, ind.categories)
        values = ind.codes
    else:
        cat = None
        values = ind.values
    if is_extension_array_dtype(ind):
        return (None, dumps(ind))
    header = (type(ind), {k: getattr(ind, k, None) for k in ind._attributes}, values.dtype, cat)
    bytes = pnp.compress(pnp.serialize(values), values.dtype)
    return (header, bytes)
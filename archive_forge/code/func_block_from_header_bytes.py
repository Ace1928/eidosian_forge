from functools import partial
import pickle
import numpy as np
import pandas as pd
from pandas.core.internals import create_block_manager_from_blocks, make_block
from . import numpy as pnp
from .core import Interface
from .encode import Encode
from .utils import extend, framesplit, frame
def block_from_header_bytes(header, bytes):
    placement, dtype, shape, (extension_type, extension_values) = header
    if extension_type == 'other':
        values = pickle.loads(bytes)
    else:
        values = pnp.deserialize(pnp.decompress(bytes, dtype), dtype, copy=True).reshape(shape)
    if extension_type == 'categorical_type':
        values = pd.Categorical.from_codes(values, extension_values[1], ordered=extension_values[0])
    elif extension_type == 'datetime64_tz_type':
        tz_info = extension_values[0]
        values = pd.DatetimeIndex(values).tz_localize('utc').tz_convert(tz_info)
    return make_block(values, placement=placement)
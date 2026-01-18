from functools import partial
import pickle
import numpy as np
import pandas as pd
from pandas.core.internals import create_block_manager_from_blocks, make_block
from . import numpy as pnp
from .core import Interface
from .encode import Encode
from .utils import extend, framesplit, frame
def is_extension_array(x):
    return False
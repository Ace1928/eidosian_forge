import json
import shutil
import tempfile
import unittest
import numpy as np
import tree
from keras.src import backend
from keras.src import ops
from keras.src import utils
from keras.src.backend.common import is_float_dtype
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.global_state import clear_session
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.models import Model
from keras.src.utils import traceback_utils
from keras.src.utils.shape_utils import map_shape_structure
def assertAlmostEqual(self, x1, x2, decimal=3, msg=None):
    if not isinstance(x1, np.ndarray):
        x1 = backend.convert_to_numpy(x1)
    if not isinstance(x2, np.ndarray):
        x2 = backend.convert_to_numpy(x2)
    np.testing.assert_almost_equal(x1, x2, decimal=decimal)
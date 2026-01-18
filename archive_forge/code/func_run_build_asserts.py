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
def run_build_asserts(layer):
    self.assertTrue(layer.built)
    if expected_num_trainable_weights is not None:
        self.assertLen(layer.trainable_weights, expected_num_trainable_weights, msg='Unexpected number of trainable_weights')
    if expected_num_non_trainable_weights is not None:
        self.assertLen(layer.non_trainable_weights, expected_num_non_trainable_weights, msg='Unexpected number of non_trainable_weights')
    if expected_num_non_trainable_variables is not None:
        self.assertLen(layer.non_trainable_variables, expected_num_non_trainable_variables, msg='Unexpected number of non_trainable_variables')
    if expected_num_seed_generators is not None:
        self.assertLen(layer._seed_generators, expected_num_seed_generators, msg='Unexpected number of _seed_generators')
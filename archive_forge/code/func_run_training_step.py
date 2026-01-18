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
def run_training_step(layer, input_data, output_data):

    class TestModel(Model):

        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def call(self, x):
            return self.layer(x)
    model = TestModel(layer)
    data = (input_data, output_data)
    if backend.backend() == 'torch':
        data = tree.map_structure(backend.convert_to_numpy, data)

    def data_generator():
        while True:
            yield data
    jit_compile = 'auto'
    if backend.backend() == 'tensorflow' and input_sparse:
        jit_compile = False
    model.compile(optimizer='sgd', loss='mse', jit_compile=jit_compile)
    model.fit(data_generator(), steps_per_epoch=1, verbose=0)
import keras_tuner
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks as blocks_module
from autokeras import keras_layers
from autokeras import nodes as nodes_module
from autokeras.engine import head as head_module
from autokeras.engine import serializable
from autokeras.utils import io_utils
def set_fit_args(self, validation_split, epochs=None):
    self.epochs = epochs
    if self.epochs is None:
        self.epochs = 1
    self.num_samples = self.inputs[0].num_samples * (1 - validation_split)
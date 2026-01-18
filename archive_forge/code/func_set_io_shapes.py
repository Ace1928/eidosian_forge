import keras_tuner
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks as blocks_module
from autokeras import keras_layers
from autokeras import nodes as nodes_module
from autokeras.engine import head as head_module
from autokeras.engine import serializable
from autokeras.utils import io_utils
def set_io_shapes(self, shapes):
    for node, shape in zip(self.inputs, nest.flatten(shapes[0])):
        node.shape = tuple(shape[1:])
    for node, shape in zip(self.outputs, nest.flatten(shapes[1])):
        node.in_blocks[0].shape = tuple(shape[1:])
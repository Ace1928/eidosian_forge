import keras_tuner
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks as blocks_module
from autokeras import keras_layers
from autokeras import nodes as nodes_module
from autokeras.engine import head as head_module
from autokeras.engine import serializable
from autokeras.utils import io_utils
def feature_encoding_input(block):
    """Fetch the column_types and column_names.

    The values are fetched for FeatureEncoding from StructuredDataInput.
    """
    if not isinstance(block.inputs[0], nodes_module.StructuredDataInput):
        raise TypeError('CategoricalToNumerical can only be used with StructuredDataInput.')
    block.column_types = block.inputs[0].column_types
    block.column_names = block.inputs[0].column_names
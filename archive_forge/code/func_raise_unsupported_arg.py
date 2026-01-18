import types
from keras.src.distribution import distribution_lib
from keras.src.trainers.data_adapters import array_data_adapter
from keras.src.trainers.data_adapters import py_dataset_adapter
from keras.src.trainers.data_adapters.array_data_adapter import ArrayDataAdapter
from keras.src.trainers.data_adapters.generator_data_adapter import (
from keras.src.trainers.data_adapters.py_dataset_adapter import PyDatasetAdapter
from keras.src.trainers.data_adapters.tf_dataset_adapter import TFDatasetAdapter
from keras.src.trainers.data_adapters.torch_data_loader_adapter import (
def raise_unsupported_arg(arg_name, arg_description, input_type):
    raise ValueError(f'When providing `x` as a {input_type}, `{arg_name}` should not be passed. Instead, {arg_description} should be included as part of the {input_type}.')
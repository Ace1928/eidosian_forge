import types
from keras.src.distribution import distribution_lib
from keras.src.trainers.data_adapters import array_data_adapter
from keras.src.trainers.data_adapters import py_dataset_adapter
from keras.src.trainers.data_adapters.array_data_adapter import ArrayDataAdapter
from keras.src.trainers.data_adapters.generator_data_adapter import (
from keras.src.trainers.data_adapters.py_dataset_adapter import PyDatasetAdapter
from keras.src.trainers.data_adapters.tf_dataset_adapter import TFDatasetAdapter
from keras.src.trainers.data_adapters.torch_data_loader_adapter import (
def is_tf_dataset(x):
    if hasattr(x, '__class__'):
        for parent in x.__class__.__mro__:
            if parent.__name__ in ('DatasetV2', 'DistributedDataset') and 'tensorflow.python.' in str(parent.__module__):
                return True
    return False
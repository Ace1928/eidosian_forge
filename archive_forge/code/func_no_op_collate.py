import math
import numpy as np
import tree
from keras.src import backend
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter
from keras.src.utils.dataset_utils import is_torch_tensor
from keras.src.utils.nest import lists_to_tuples
def no_op_collate(batch):
    return batch
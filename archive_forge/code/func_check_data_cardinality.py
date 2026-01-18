import math
import numpy as np
import tree
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.utils.dataset_utils import is_torch_tensor
def check_data_cardinality(data):
    num_samples = set((int(i.shape[0]) for i in tree.flatten(data)))
    if len(num_samples) > 1:
        msg = 'Data cardinality is ambiguous. Make sure all arrays contain the same number of samples.'
        for label, single_data in zip(['x', 'y', 'sample_weight'], data):
            sizes = ', '.join((str(i.shape[0]) for i in tree.flatten(single_data)))
            msg += f"'{label}' sizes: {sizes}\n"
        raise ValueError(msg)
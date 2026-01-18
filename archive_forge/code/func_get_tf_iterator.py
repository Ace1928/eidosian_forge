import itertools
import numpy as np
import tree
from keras.src import backend
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter
def get_tf_iterator():
    for batch in self.generator:
        batch = tree.map_structure(convert_to_tf, batch)
        yield batch
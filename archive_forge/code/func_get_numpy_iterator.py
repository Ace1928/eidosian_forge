import numpy as np
import tree
from keras.src import backend
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter
def get_numpy_iterator(self):
    for batch in self._dataloader:
        yield tuple(tree.map_structure(lambda x: np.asarray(x.cpu()), batch))
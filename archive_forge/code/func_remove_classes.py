import collections
import multiprocessing
import os
import threading
import warnings
import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.trainers.data_adapters.py_dataset_adapter import PyDataset
from keras.src.utils import image_utils
from keras.src.utils import io_utils
from keras.src.utils.module_utils import scipy
def remove_classes(labels, classes):
    if isinstance(labels, (list, tuple)):
        labels = [cls for cls in labels if cls in classes]
        return labels or None
    elif isinstance(labels, str):
        return labels if labels in classes else None
    else:
        raise TypeError('Expect string, list or tuple but found {} in {} column '.format(type(labels), y_col))
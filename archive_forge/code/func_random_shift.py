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
@keras_export('keras._legacy.preprocessing.image.random_shift')
def random_shift(x, wrg, hrg, row_axis=1, col_axis=2, channel_axis=0, fill_mode='nearest', cval=0.0, interpolation_order=1):
    """DEPRECATED."""
    h, w = (x.shape[row_axis], x.shape[col_axis])
    tx = np.random.uniform(-hrg, hrg) * h
    ty = np.random.uniform(-wrg, wrg) * w
    x = apply_affine_transform(x, tx=tx, ty=ty, row_axis=row_axis, col_axis=col_axis, channel_axis=channel_axis, fill_mode=fill_mode, cval=cval, order=interpolation_order)
    return x
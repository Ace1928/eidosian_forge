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
@keras_export('keras._legacy.preprocessing.image.random_zoom')
def random_zoom(x, zoom_range, row_axis=1, col_axis=2, channel_axis=0, fill_mode='nearest', cval=0.0, interpolation_order=1):
    """DEPRECATED."""
    if len(zoom_range) != 2:
        raise ValueError(f'`zoom_range` should be a tuple or list of two floats. Received: {zoom_range}')
    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = (1, 1)
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    x = apply_affine_transform(x, zx=zx, zy=zy, row_axis=row_axis, col_axis=col_axis, channel_axis=channel_axis, fill_mode=fill_mode, cval=cval, order=interpolation_order)
    return x
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
def flow_from_directory(self, directory, target_size=(256, 256), color_mode='rgb', classes=None, class_mode='categorical', batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png', follow_links=False, subset=None, interpolation='nearest', keep_aspect_ratio=False):
    return DirectoryIterator(directory, self, target_size=target_size, color_mode=color_mode, keep_aspect_ratio=keep_aspect_ratio, classes=classes, class_mode=class_mode, data_format=self.data_format, batch_size=batch_size, shuffle=shuffle, seed=seed, save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format, follow_links=follow_links, subset=subset, interpolation=interpolation, dtype=self.dtype)
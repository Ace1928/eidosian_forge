import os
import random
import time
import warnings
from multiprocessing.pool import ThreadPool
import numpy as np
from keras.src.api_export import keras_export
from keras.src.utils import io_utils
from keras.src.utils.module_utils import tensorflow as tf
def is_torch_dataset(dataset):
    if hasattr(dataset, '__class__'):
        for parent in dataset.__class__.__mro__:
            if parent.__name__ == 'Dataset' and str(parent.__module__).startswith('torch.utils.data'):
                return True
    return False
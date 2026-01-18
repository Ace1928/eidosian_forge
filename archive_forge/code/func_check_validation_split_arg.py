import multiprocessing
import os
import random
import time
import warnings
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src.utils import io_utils
from tensorflow.python.util.tf_export import keras_export
def check_validation_split_arg(validation_split, subset, shuffle, seed):
    """Raise errors in case of invalid argument values.

    Args:
      validation_split: float between 0 and 1, fraction of data to reserve for
        validation.
      subset: One of "training", "validation" or "both". Only used if
        `validation_split` is set.
      shuffle: Whether to shuffle the data. Either True or False.
      seed: random seed for shuffling and transformations.
    """
    if validation_split and (not 0 < validation_split < 1):
        raise ValueError(f'`validation_split` must be between 0 and 1, received: {validation_split}')
    if (validation_split or subset) and (not (validation_split and subset)):
        raise ValueError('If `subset` is set, `validation_split` must be set, and inversely.')
    if subset not in ('training', 'validation', 'both', None):
        raise ValueError(f'`subset` must be either "training", "validation" or "both", received: {subset}')
    if validation_split and shuffle and (seed is None):
        raise ValueError('If using `validation_split` and shuffling the data, you must provide a `seed` argument, to make sure that there is no overlap between the training and validation subset.')
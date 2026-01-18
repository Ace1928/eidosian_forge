import collections
import inspect
import os
import pickle
import numpy as np
from keras_tuner.src import backend
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import base_tuner
def split_data(data, indices):
    if isinstance(data, np.ndarray):
        return data[indices]
    elif pd is not None and isinstance(data, pd.DataFrame):
        return data.iloc[indices]
    else:
        raise TypeError(f'Expected the data to be numpy.ndarray or pandas.DataFrame. Received: {data}.')
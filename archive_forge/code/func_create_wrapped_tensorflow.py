import pytest
from thinc.api import (
from thinc.compat import has_tensorflow, has_torch
def create_wrapped_tensorflow(width, dropout, nI, nO):
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.models import Sequential
    tf_model = Sequential()
    tf_model.add(Dense(width, activation='relu', input_shape=(nI,)))
    tf_model.add(Dropout(dropout))
    tf_model.add(Dense(width, activation='relu'))
    tf_model.add(Dropout(dropout))
    tf_model.add(Dense(nO, activation=None))
    return TensorFlowWrapper(tf_model)
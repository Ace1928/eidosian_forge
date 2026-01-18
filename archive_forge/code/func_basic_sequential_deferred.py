import collections
import keras.src as keras
def basic_sequential_deferred():
    """Sequential model with deferred input shape."""
    model = keras.Sequential([keras.layers.Dense(3, activation='relu'), keras.layers.Dense(2, activation='softmax')])
    return ModelFn(model, (None, 3), (None, 2))
import collections
import keras.src as keras
def functional_with_keyword_args():
    """A functional model with keyword args."""
    inputs = keras.Input(shape=(3,))
    x = keras.layers.Dense(4)(inputs)
    x = keras.layers.BatchNormalization()(x)
    outputs = keras.layers.Dense(2)(x)
    model = keras.Model(inputs, outputs, name='m', trainable=False)
    return ModelFn(model, (None, 3), (None, 2))
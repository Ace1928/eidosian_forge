import collections
import keras.src as keras
def shared_sequential():
    """Shared sequential model in a functional model."""
    inner_model = keras.Sequential([keras.layers.Conv2D(2, 3, activation='relu'), keras.layers.Conv2D(2, 3, activation='relu')])
    inputs_1 = keras.Input((5, 5, 3))
    inputs_2 = keras.Input((5, 5, 3))
    x1 = inner_model(inputs_1)
    x2 = inner_model(inputs_2)
    x = keras.layers.concatenate([x1, x2])
    outputs = keras.layers.GlobalAveragePooling2D()(x)
    model = keras.Model([inputs_1, inputs_2], outputs)
    return ModelFn(model, [(None, 5, 5, 3), (None, 5, 5, 3)], (None, 4))
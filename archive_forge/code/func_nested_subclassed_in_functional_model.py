import collections
import keras.src as keras
def nested_subclassed_in_functional_model():
    """A subclass model nested in a functional model."""
    inner_subclass_model = MySubclassModel()
    inputs = keras.Input(shape=(3,))
    x = inner_subclass_model(inputs)
    x = keras.layers.BatchNormalization()(x)
    outputs = keras.layers.Dense(2, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    return ModelFn(model, (None, 3), (None, 2))
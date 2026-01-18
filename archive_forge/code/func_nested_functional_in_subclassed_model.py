import collections
import keras.src as keras
def nested_functional_in_subclassed_model():
    """A functional model nested in a subclass model."""

    def get_functional_model():
        inputs = keras.Input(shape=(4,))
        x = keras.layers.Dense(4, activation='relu')(inputs)
        x = keras.layers.BatchNormalization()(x)
        outputs = keras.layers.Dense(2)(x)
        return keras.Model(inputs, outputs)

    class NestedFunctionalInSubclassModel(keras.Model):
        """A functional nested in subclass model."""

        def __init__(self):
            super().__init__(name='nested_functional_in_subclassed_model')
            self.dense1 = keras.layers.Dense(4, activation='relu')
            self.dense2 = keras.layers.Dense(2, activation='relu')
            self.inner_functional_model = get_functional_model()

        def call(self, inputs):
            x = self.dense1(inputs)
            x = self.inner_functional_model(x)
            return self.dense2(x)
    return ModelFn(NestedFunctionalInSubclassModel(), (None, 3), (None, 2))
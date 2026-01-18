import collections
import keras.src as keras
def shared_layer_subclassed_model():
    """Shared layer in a subclass model."""

    class SharedLayerSubclassModel(keras.Model):
        """A subclass model with shared layers."""

        def __init__(self):
            super().__init__(name='shared_layer_subclass_model')
            self.dense = keras.layers.Dense(3, activation='relu')
            self.dp = keras.layers.Dropout(0.5)
            self.bn = keras.layers.BatchNormalization()

        def call(self, inputs):
            x = self.dense(inputs)
            x = self.dp(x)
            x = self.bn(x)
            return self.dense(x)
    return ModelFn(SharedLayerSubclassModel(), (None, 3), (None, 3))
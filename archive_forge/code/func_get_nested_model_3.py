import keras.src as keras
from keras.src.testing_infra import test_utils
def get_nested_model_3(input_dim, num_classes):
    inputs = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(32, activation='relu')(inputs)
    x = keras.layers.BatchNormalization()(x)

    class Inner(keras.Model):

        def __init__(self):
            super().__init__()
            self.dense1 = keras.layers.Dense(32, activation='relu')
            self.dense2 = keras.layers.Dense(5, activation='relu')
            self.bn = keras.layers.BatchNormalization()

        def call(self, inputs):
            x = self.dense1(inputs)
            x = self.dense2(x)
            return self.bn(x)
    test_model = Inner()
    x = test_model(x)
    outputs = keras.layers.Dense(num_classes)(x)
    return keras.Model(inputs, outputs, name='nested_model_3')
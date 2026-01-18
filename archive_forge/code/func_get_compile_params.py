from tensorflow.keras import layers, models
from tensorflow.keras.datasets import boston_housing
from tune_tensorflow import KerasTrainingSpec
def get_compile_params(self):
    return dict(optimizer='rmsprop', loss='mse', metrics=['mae'])
import tree
from keras.src import backend
from keras.src import layers
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.saving import saving_lib
from keras.src.saving import serialization_lib
from keras.src.utils import backend_utils
from keras.src.utils.module_utils import tensorflow as tf
from keras.src.utils.naming import auto_name
def get_encoded_features(self):
    self._check_if_adapted()
    if self.encoded_features is None:
        preprocessed_features = self._preprocess_features(self.inputs)
        crossed_features = self._cross_features(preprocessed_features)
        merged_features = self._merge_features(preprocessed_features, crossed_features)
        self.encoded_features = merged_features
    return self.encoded_features
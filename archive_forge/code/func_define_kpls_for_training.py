import random
import tempfile
import tensorflow.compat.v2 as tf
import keras.src as keras
from keras.src.layers.preprocessing import string_lookup
def define_kpls_for_training(self, use_adapt):
    """Function that defines KPL used for unit tests of tf.distribute.

        Args:
          use_adapt: if adapt will be called. False means there will be
            precomputed statistics.

        Returns:
          feature_mapper: a simple keras model with one keras StringLookup layer
          which maps feature to index.
          label_mapper: similar to feature_mapper, but maps label to index.

        """
    if use_adapt:
        feature_lookup_layer = string_lookup.StringLookup(num_oov_indices=1)
        feature_lookup_layer.adapt(self.FEATURE_VOCAB)
        label_lookup_layer = string_lookup.StringLookup(num_oov_indices=0, mask_token=None)
        label_lookup_layer.adapt(self.LABEL_VOCAB)
    else:
        feature_lookup_layer = string_lookup.StringLookup(vocabulary=self.FEATURE_VOCAB, num_oov_indices=1)
        label_lookup_layer = string_lookup.StringLookup(vocabulary=self.LABEL_VOCAB, num_oov_indices=0, mask_token=None)
    raw_feature_input = keras.layers.Input(shape=(3,), dtype=tf.string, name='feature', ragged=True)
    feature_id_input = feature_lookup_layer(raw_feature_input)
    feature_mapper = keras.Model({'features': raw_feature_input}, feature_id_input)
    raw_label_input = keras.layers.Input(shape=(1,), dtype=tf.string, name='label')
    label_id_input = label_lookup_layer(raw_label_input)
    label_mapper = keras.Model({'label': raw_label_input}, label_id_input)
    return (feature_mapper, label_mapper)
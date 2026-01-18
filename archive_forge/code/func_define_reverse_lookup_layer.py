import random
import tempfile
import tensorflow.compat.v2 as tf
import keras.src as keras
from keras.src.layers.preprocessing import string_lookup
def define_reverse_lookup_layer(self):
    """Create string reverse lookup layer for serving."""
    label_inverse_lookup_layer = string_lookup.StringLookup(num_oov_indices=0, mask_token=None, vocabulary=self.LABEL_VOCAB, invert=True)
    return label_inverse_lookup_layer
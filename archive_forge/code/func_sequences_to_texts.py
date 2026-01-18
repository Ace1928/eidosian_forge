import collections
import hashlib
import json
import warnings
import numpy as np
from tensorflow.python.util.tf_export import keras_export
def sequences_to_texts(self, sequences):
    """Transforms each sequence into a list of text.

        Only top `num_words-1` most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.

        Args:
            sequences: A list of sequences (list of integers).

        Returns:
            A list of texts (strings)
        """
    return list(self.sequences_to_texts_generator(sequences))
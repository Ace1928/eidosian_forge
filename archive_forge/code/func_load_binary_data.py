import logging
import numpy as np
from numpy import ones, vstack, float32 as REAL
import gensim.models._fasttext_bin
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors, prep_vectors
from gensim import utils
from gensim.utils import deprecated
from gensim.models import keyedvectors  # noqa: E402
@utils.deprecated('use load_facebook_vectors (to use pretrained embeddings) or load_facebook_model (to continue training with the loaded full model, more RAM) instead')
def load_binary_data(self, encoding='utf8'):
    """Load data from a binary file created by Facebook's native FastText.

        Parameters
        ----------
        encoding : str, optional
            Specifies the encoding.

        """
    m = _load_fasttext_format(self.file_name, encoding=encoding)
    for attr, val in m.__dict__.items():
        setattr(self, attr, val)
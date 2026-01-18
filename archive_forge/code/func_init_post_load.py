import logging
import numpy as np
from numpy import ones, vstack, float32 as REAL
import gensim.models._fasttext_bin
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors, prep_vectors
from gensim import utils
from gensim.utils import deprecated
from gensim.models import keyedvectors  # noqa: E402
def init_post_load(self, fb_vectors):
    """Perform initialization after loading a native Facebook model.

        Expects that the vocabulary (self.key_to_index) has already been initialized.

        Parameters
        ----------
        fb_vectors : np.array
            A matrix containing vectors for all the entities, including words
            and ngrams.  This comes directly from the binary model.
            The order of the vectors must correspond to the indices in
            the vocabulary.

        """
    vocab_words = len(self)
    assert fb_vectors.shape[0] == vocab_words + self.bucket, 'unexpected number of vectors'
    assert fb_vectors.shape[1] == self.vector_size, 'unexpected vector dimensionality'
    self.vectors_vocab = np.array(fb_vectors[:vocab_words, :])
    self.vectors_ngrams = np.array(fb_vectors[vocab_words:, :])
    self.recalc_char_ngram_buckets()
    self.adjust_vectors()
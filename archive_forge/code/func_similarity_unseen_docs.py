import logging
import os
from collections import namedtuple, defaultdict
from collections.abc import Iterable
from timeit import default_timer
from dataclasses import dataclass
from numpy import zeros, float32 as REAL, vstack, integer, dtype
import numpy as np
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.utils import deprecated
from gensim.models import Word2Vec, FAST_VERSION  # noqa: F401
from gensim.models.keyedvectors import KeyedVectors, pseudorandom_weak_vector
def similarity_unseen_docs(self, doc_words1, doc_words2, alpha=None, min_alpha=None, epochs=None):
    """Compute cosine similarity between two post-bulk out of training documents.

        Parameters
        ----------
        model : :class:`~gensim.models.doc2vec.Doc2Vec`
            An instance of a trained `Doc2Vec` model.
        doc_words1 : list of str
            Input document.
        doc_words2 : list of str
            Input document.
        alpha : float, optional
            The initial learning rate.
        min_alpha : float, optional
            Learning rate will linearly drop to `min_alpha` as training progresses.
        epochs : int, optional
            Number of epoch to train the new document.

        Returns
        -------
        float
            The cosine similarity between `doc_words1` and `doc_words2`.

        """
    d1 = self.infer_vector(doc_words=doc_words1, alpha=alpha, min_alpha=min_alpha, epochs=epochs)
    d2 = self.infer_vector(doc_words=doc_words2, alpha=alpha, min_alpha=min_alpha, epochs=epochs)
    return np.dot(matutils.unitvec(d1), matutils.unitvec(d2))
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
def reset_from(self, other_model):
    """Copy shareable data structures from another (possibly pre-trained) model.

        This specifically causes some structures to be shared, so is limited to
        structures (like those rleated to the known word/tag vocabularies) that
        won't change during training or thereafter. Beware vocabulary edits/updates
        to either model afterwards: the partial sharing and out-of-band modification
        may leave the other model in a broken state.

        Parameters
        ----------
        other_model : :class:`~gensim.models.doc2vec.Doc2Vec`
            Other model whose internal data structures will be copied over to the current object.

        """
    self.wv.key_to_index = other_model.wv.key_to_index
    self.wv.index_to_key = other_model.wv.index_to_key
    self.wv.expandos = other_model.wv.expandos
    self.cum_table = other_model.cum_table
    self.corpus_count = other_model.corpus_count
    self.dv.key_to_index = other_model.dv.key_to_index
    self.dv.index_to_key = other_model.dv.index_to_key
    self.dv.expandos = other_model.dv.expandos
    self.init_weights()
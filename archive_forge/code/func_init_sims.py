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
@deprecated('Gensim 4.0.0 implemented internal optimizations that make calls to init_sims() unnecessary. init_sims() is now obsoleted and will be completely removed in future versions. See https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4')
def init_sims(self, replace=False):
    """
        Precompute L2-normalized vectors. Obsoleted.

        If you need a single unit-normalized vector for some key, call
        :meth:`~gensim.models.keyedvectors.KeyedVectors.get_vector` instead:
        ``doc2vec_model.dv.get_vector(key, norm=True)``.

        To refresh norms after you performed some atypical out-of-band vector tampering,
        call `:meth:`~gensim.models.keyedvectors.KeyedVectors.fill_norms()` instead.

        Parameters
        ----------
        replace : bool
            If True, forget the original trained vectors and only keep the normalized ones.
            You lose information if you do this.

        """
    self.dv.init_sims(replace=replace)
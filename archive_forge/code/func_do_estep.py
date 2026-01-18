import logging
from itertools import chain
from copy import deepcopy
from shutil import copyfile
from os.path import isfile
from os import remove
import numpy as np  # for arrays, array broadcasting etc.
from scipy.special import gammaln  # gamma function utils
from gensim import utils
from gensim.models import LdaModel
from gensim.models.ldamodel import LdaState
from gensim.matutils import dirichlet_expectation, mean_absolute_difference
from gensim.corpora import MmCorpus
def do_estep(self, chunk, author2doc, doc2author, rhot, state=None, chunk_doc_idx=None):
    """Performs inference (E-step) on a chunk of documents, and accumulate the collected sufficient statistics.

        Parameters
        ----------
        chunk : iterable of list of (int, float)
            Corpus in BoW format.
        author2doc : dict of (str, list of int), optional
            A dictionary where keys are the names of authors and values are lists of document IDs that the author
            contributes to.
        doc2author : dict of (int, list of str), optional
            A dictionary where the keys are document IDs and the values are lists of author names.
        rhot : float
            Value of rho for conducting inference on documents.
        state : int, optional
            Initializes the state for a new E iteration.
        chunk_doc_idx : numpy.ndarray, optional
            Assigns the value for document index.

        Returns
        -------
        float
            Value of gamma for training of model.

        """
    if state is None:
        state = self.state
    gamma, sstats = self.inference(chunk, author2doc, doc2author, rhot, collect_sstats=True, chunk_doc_idx=chunk_doc_idx)
    state.sstats += sstats
    state.numdocs += len(chunk)
    return gamma
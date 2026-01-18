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
def extend_corpus(self, corpus):
    """Add new documents from `corpus` to `self.corpus`.

        If serialization is used, then the entire corpus (`self.corpus`) is re-serialized and the new documents
        are added in the process. If serialization is not used, the corpus, as a list of documents, is simply extended.

        Parameters
        ----------
        corpus : iterable of list of (int, float)
            Corpus in BoW format

        Raises
        ------
        AssertionError
            If serialized == False and corpus isn't list.

        """
    if self.serialized:
        if isinstance(corpus, MmCorpus):
            assert self.corpus.input != corpus.input, 'Input corpus cannot have the same file path as the model corpus (serialization_path).'
        corpus_chain = chain(self.corpus, corpus)
        copyfile(self.serialization_path, self.serialization_path + '.tmp')
        self.corpus.input = self.serialization_path + '.tmp'
        MmCorpus.serialize(self.serialization_path, corpus_chain)
        self.corpus = MmCorpus(self.serialization_path)
        remove(self.serialization_path + '.tmp')
    else:
        assert isinstance(corpus, list), 'If serialized == False, all input corpora must be lists.'
        self.corpus.extend(corpus)
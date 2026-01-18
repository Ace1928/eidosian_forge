import logging
import itertools
import os
import heapq
import warnings
import numpy
import scipy.sparse
from gensim import interfaces, utils, matutils
def similarity_by_id(self, docpos):
    """Get similarity of a document specified by its index position `docpos`.

        Parameters
        ----------
        docpos : int
            Document position in the index.

        Return
        ------
        :class:`numpy.ndarray` or :class:`scipy.sparse.csr_matrix`
            Similarities of the given document against this index.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.corpora.textcorpus import TextCorpus
            >>> from gensim.test.utils import datapath
            >>> from gensim.similarities import Similarity
            >>>
            >>> corpus = TextCorpus(datapath('testcorpus.txt'))
            >>> index = Similarity('temp', corpus, num_features=400)
            >>> similarities = index.similarity_by_id(1)

        """
    query = self.vector_by_id(docpos)
    norm, self.norm = (self.norm, False)
    result = self[query]
    self.norm = norm
    return result
from __future__ import with_statement
import logging
import math
from gensim import utils
import numpy as np
import scipy.sparse
from scipy.stats import entropy
from scipy.linalg import get_blas_funcs, triu
from scipy.linalg.lapack import get_lapack_funcs
from scipy.special import psi  # gamma function utils
def fake_headers(self, num_docs, num_terms, num_nnz):
    """Write "fake" headers to file, to be rewritten once we've scanned the entire corpus.

        Parameters
        ----------
        num_docs : int
            Number of documents in corpus.
        num_terms : int
            Number of term in corpus.
        num_nnz : int
            Number of non-zero elements in corpus.

        """
    stats = '%i %i %i' % (num_docs, num_terms, num_nnz)
    if len(stats) > 50:
        raise ValueError('Invalid stats: matrix too large!')
    self.fout.seek(len(MmWriter.HEADER_LINE))
    self.fout.write(utils.to_utf8(stats))
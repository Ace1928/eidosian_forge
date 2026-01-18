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
def construct_doc2author(corpus, author2doc):
    """Create a mapping from document IDs to author IDs.

    Parameters
    ----------
    corpus: iterable of list of (int, float)
        Corpus in BoW format.
    author2doc: dict of (str, list of int)
        Mapping of authors to documents.

    Returns
    -------
    dict of (int, list of str)
        Document to Author mapping.

    """
    doc2author = {}
    for d, _ in enumerate(corpus):
        author_ids = []
        for a, a_doc_ids in author2doc.items():
            if d in a_doc_ids:
                author_ids.append(a)
        doc2author[d] = author_ids
    return doc2author
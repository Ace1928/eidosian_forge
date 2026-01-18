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
def rollback_new_author_chages():
    self.state.gamma = self.state.gamma[0:-1]
    del self.author2doc[new_author_name]
    a_id = self.author2id[new_author_name]
    del self.id2author[a_id]
    del self.author2id[new_author_name]
    for new_doc_id in corpus_doc_idx:
        del self.doc2author[new_doc_id]
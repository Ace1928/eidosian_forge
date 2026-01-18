import numpy as np
import scipy.sparse as sp
from multiprocessing.pool import ThreadPool
from functools import partial
from . import utils
from . import tokenizers
from parlai.utils.logging import logger
def get_doc_index(self, doc_id):
    """
        Convert doc_id --> doc_index.
        """
    return self.doc_dict[0][doc_id] if self.doc_dict else doc_id
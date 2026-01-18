import os
import pickle
import time
from typing import Iterable, List, Optional, Tuple
import numpy as np
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import BatchEncoding
from ...utils import cached_file, is_datasets_available, is_faiss_available, logging, requires_backends, strtobool
from .configuration_rag import RagConfig
from .tokenization_rag import RagTokenizer
def get_doc_dicts(self, doc_ids: np.ndarray) -> List[dict]:
    return [self.dataset[doc_ids[i].tolist()] for i in range(doc_ids.shape[0])]
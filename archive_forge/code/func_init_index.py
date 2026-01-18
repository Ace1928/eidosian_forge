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
def init_index(self):
    if not self.is_initialized():
        logger.info(f'Loading index from {self.index_path}')
        self.dataset.load_faiss_index('embeddings', file=self.index_path)
        self._index_initialized = True
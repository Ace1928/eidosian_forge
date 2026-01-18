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
def cat_input_and_doc(doc_title, doc_text, input_string, prefix):
    if doc_title.startswith('"'):
        doc_title = doc_title[1:]
    if doc_title.endswith('"'):
        doc_title = doc_title[:-1]
    if prefix is None:
        prefix = ''
    out = (prefix + doc_title + self.config.title_sep + doc_text + self.config.doc_sep + input_string).replace('  ', ' ')
    return out
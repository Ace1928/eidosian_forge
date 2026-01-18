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
def retrieve(self, question_hidden_states: np.ndarray, n_docs: int) -> Tuple[np.ndarray, List[dict]]:
    """
        Retrieves documents for specified `question_hidden_states`.

        Args:
            question_hidden_states (`np.ndarray` of shape `(batch_size, vector_size)`):
                A batch of query vectors to retrieve with.
            n_docs (`int`):
                The number of docs retrieved per query.

        Return:
            `Tuple[np.ndarray, np.ndarray, List[dict]]`: A tuple with the following objects:

            - **retrieved_doc_embeds** (`np.ndarray` of shape `(batch_size, n_docs, dim)`) -- The retrieval embeddings
              of the retrieved docs per query.
            - **doc_ids** (`np.ndarray` of shape `(batch_size, n_docs)`) -- The ids of the documents in the index
            - **doc_dicts** (`List[dict]`): The `retrieved_doc_embeds` examples per query.
        """
    doc_ids, retrieved_doc_embeds = self._main_retrieve(question_hidden_states, n_docs)
    return (retrieved_doc_embeds, doc_ids, self.index.get_doc_dicts(doc_ids))
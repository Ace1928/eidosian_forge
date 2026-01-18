import importlib.util
import os
import tempfile
from pathlib import PurePath
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Union
import fsspec
import numpy as np
from .utils import logging
from .utils import tqdm as hf_tqdm
def get_nearest_examples_batch(self, index_name: str, queries: Union[List[str], np.array], k: int=10, **kwargs) -> BatchedNearestExamplesResults:
    """Find the nearest examples in the dataset to the query.

        Args:
            index_name (`str`):
                The `index_name`/identifier of the index.
            queries (`Union[List[str], np.ndarray]`):
                The queries as a list of strings if `index_name` is a text index or as a numpy array if `index_name` is a vector index.
            k (`int`):
                The number of examples to retrieve per query.

        Returns:
            `(total_scores, total_examples)`:
                A tuple of `(total_scores, total_examples)` where:
                - **total_scores** (`List[List[float]`): the retrieval scores from either FAISS (`IndexFlatL2` by default) or ElasticSearch of the retrieved examples per query
                - **total_examples** (`List[dict]`): the retrieved examples per query
        """
    self._check_index_is_initialized(index_name)
    total_scores, total_indices = self.search_batch(index_name, queries, k, **kwargs)
    total_scores = [scores_i[:len([i for i in indices_i if i >= 0])] for scores_i, indices_i in zip(total_scores, total_indices)]
    total_samples = [self[[i for i in indices if i >= 0]] for indices in total_indices]
    return BatchedNearestExamplesResults(total_scores, total_samples)
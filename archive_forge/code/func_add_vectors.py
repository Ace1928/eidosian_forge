import importlib.util
import os
import tempfile
from pathlib import PurePath
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Union
import fsspec
import numpy as np
from .utils import logging
from .utils import tqdm as hf_tqdm
def add_vectors(self, vectors: Union[np.array, 'Dataset'], column: Optional[str]=None, batch_size: int=1000, train_size: Optional[int]=None, faiss_verbose: Optional[bool]=None):
    """
        Add vectors to the index.
        If the arrays are inside a certain column, you can specify it using the `column` argument.
        """
    import faiss
    if self.faiss_index is None:
        size = len(vectors[0]) if column is None else len(vectors[0][column])
        if self.string_factory is not None:
            if self.metric_type is None:
                index = faiss.index_factory(size, self.string_factory)
            else:
                index = faiss.index_factory(size, self.string_factory, self.metric_type)
        elif self.metric_type is None:
            index = faiss.IndexFlat(size)
        else:
            index = faiss.IndexFlat(size, self.metric_type)
        self.faiss_index = self._faiss_index_to_device(index, self.device)
        logger.info(f'Created faiss index of type {type(self.faiss_index)}')
    if faiss_verbose is not None:
        self.faiss_index.verbose = faiss_verbose
        if hasattr(self.faiss_index, 'index') and self.faiss_index.index is not None:
            self.faiss_index.index.verbose = faiss_verbose
        if hasattr(self.faiss_index, 'quantizer') and self.faiss_index.quantizer is not None:
            self.faiss_index.quantizer.verbose = faiss_verbose
        if hasattr(self.faiss_index, 'clustering_index') and self.faiss_index.clustering_index is not None:
            self.faiss_index.clustering_index.verbose = faiss_verbose
    if train_size is not None:
        train_vecs = vectors[:train_size] if column is None else vectors[:train_size][column]
        logger.info(f'Training the index with the first {len(train_vecs)} vectors')
        self.faiss_index.train(train_vecs)
    else:
        logger.info('Ignored the training step of the faiss index as `train_size` is None.')
    logger.info(f'Adding {len(vectors)} vectors to the faiss index')
    for i in hf_tqdm(range(0, len(vectors), batch_size)):
        vecs = vectors[i:i + batch_size] if column is None else vectors[i:i + batch_size][column]
        self.faiss_index.add(vecs)
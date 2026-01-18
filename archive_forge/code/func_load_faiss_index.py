import importlib.util
import os
import tempfile
from pathlib import PurePath
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Union
import fsspec
import numpy as np
from .utils import logging
from .utils import tqdm as hf_tqdm
def load_faiss_index(self, index_name: str, file: Union[str, PurePath], device: Optional[Union[int, List[int]]]=None, storage_options: Optional[Dict]=None):
    """Load a FaissIndex from disk.

        If you want to do additional configurations, you can have access to the faiss index object by doing
        `.get_index(index_name).faiss_index` to make it fit your needs.

        Args:
            index_name (`str`): The index_name/identifier of the index. This is the index_name that is used to
                call `.get_nearest` or `.search`.
            file (`str`): The path to the serialized faiss index on disk or remote URI (e.g. `"s3://my-bucket/index.faiss"`).
            device (Optional `Union[int, List[int]]`): If positive integer, this is the index of the GPU to use. If negative integer, use all GPUs.
                If a list of positive integers is passed in, run only on those GPUs. By default it uses the CPU.
            storage_options (`dict`, *optional*):
                Key/value pairs to be passed on to the file-system backend, if any.

                <Added version="2.11.0"/>

        """
    index = FaissIndex.load(file, device=device, storage_options=storage_options)
    if index.faiss_index.ntotal != len(self):
        raise ValueError(f"Index size should match Dataset size, but Index '{index_name}' at {file} has {index.faiss_index.ntotal} elements while the dataset has {len(self)} examples.")
    self._indexes[index_name] = index
    logger.info(f'Loaded FaissIndex {index_name} from {file}')
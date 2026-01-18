import importlib
import shutil
import threading
import warnings
from typing import List
import fsspec
import fsspec.asyn
from fsspec.implementations.local import LocalFileSystem
from ..utils.deprecation_utils import deprecated
from . import compression
@deprecated('This function is deprecated and will be removed in a future version. Please use `fsspec.core.strip_protocol` instead.')
def extract_path_from_uri(dataset_path: str) -> str:
    """
    Preprocesses `dataset_path` and removes remote filesystem (e.g. removing `s3://`).

    Args:
        dataset_path (`str`):
            Path (e.g. `dataset/train`) or remote uri (e.g. `s3://my-bucket/dataset/train`) of the dataset directory.
    """
    if '://' in dataset_path:
        dataset_path = dataset_path.split('://')[1]
    return dataset_path
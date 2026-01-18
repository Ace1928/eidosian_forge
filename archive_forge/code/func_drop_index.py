import importlib.util
import os
import tempfile
from pathlib import PurePath
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Union
import fsspec
import numpy as np
from .utils import logging
from .utils import tqdm as hf_tqdm
def drop_index(self, index_name: str):
    """Drop the index with the specified column.

        Args:
            index_name (`str`):
                The `index_name`/identifier of the index.
        """
    del self._indexes[index_name]
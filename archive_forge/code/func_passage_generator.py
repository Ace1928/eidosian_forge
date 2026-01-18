import importlib.util
import os
import tempfile
from pathlib import PurePath
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Union
import fsspec
import numpy as np
from .utils import logging
from .utils import tqdm as hf_tqdm
def passage_generator():
    if column is not None:
        for i, example in enumerate(documents):
            yield {'text': example[column], '_id': i}
    else:
        for i, example in enumerate(documents):
            yield {'text': example, '_id': i}
import os
from typing import BinaryIO, Optional, Union
import numpy as np
import pyarrow.parquet as pq
from .. import Audio, Dataset, Features, Image, NamedSplit, Value, config
from ..features.features import FeatureType, _visit
from ..formatting import query_table
from ..packaged_modules import _PACKAGED_DATASETS_MODULES
from ..packaged_modules.parquet.parquet import Parquet
from ..utils import tqdm as hf_tqdm
from ..utils.typing import NestedDataStructureLike, PathLike
from .abc import AbstractDatasetReader
def set_batch_size(feature: FeatureType) -> None:
    nonlocal batch_size
    if isinstance(feature, Image):
        batch_size = min(batch_size, config.PARQUET_ROW_GROUP_SIZE_FOR_IMAGE_DATASETS)
    elif isinstance(feature, Audio):
        batch_size = min(batch_size, config.PARQUET_ROW_GROUP_SIZE_FOR_AUDIO_DATASETS)
    elif isinstance(feature, Value) and feature.dtype == 'binary':
        batch_size = min(batch_size, config.PARQUET_ROW_GROUP_SIZE_FOR_BINARY_DATASETS)
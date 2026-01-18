import itertools
import logging
import os
import pathlib
import re
from typing import (
import numpy as np
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data.block import BlockMetadata
from ray.data.datasource.partitioning import Partitioning
from ray.util.annotations import DeveloperAPI
def prefetch_file_metadata(self, fragments: List['pyarrow.dataset.ParquetFileFragment'], **ray_remote_args) -> Optional[List['pyarrow.parquet.FileMetaData']]:
    from ray.data.datasource.parquet_datasource import FRAGMENTS_PER_META_FETCH, PARALLELIZE_META_FETCH_THRESHOLD, _fetch_metadata, _fetch_metadata_serialization_wrapper, _SerializedFragment
    if len(fragments) > PARALLELIZE_META_FETCH_THRESHOLD:
        fragments = [_SerializedFragment(fragment) for fragment in fragments]
        return list(_fetch_metadata_parallel(fragments, _fetch_metadata_serialization_wrapper, FRAGMENTS_PER_META_FETCH, **ray_remote_args))
    else:
        return _fetch_metadata(fragments)
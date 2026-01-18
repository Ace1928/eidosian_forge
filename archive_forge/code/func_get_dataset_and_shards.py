import logging
import math
from pathlib import Path
import re
import numpy as np
from typing import List, Tuple, TYPE_CHECKING, Optional
import zipfile
import ray.data
from ray.rllib.offline.input_reader import InputReader
from ray.rllib.offline.io_context import IOContext
from ray.rllib.offline.json_reader import from_json_data, postprocess_actions
from ray.rllib.policy.sample_batch import concat_samples, SampleBatch, DEFAULT_POLICY_ID
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.typing import SampleBatchType
@PublicAPI
def get_dataset_and_shards(config: 'AlgorithmConfig', num_workers: int=0) -> Tuple[ray.data.Dataset, List[ray.data.Dataset]]:
    """Returns a dataset and a list of shards.

    This function uses algorithm configs to create a dataset and a list of shards.
    The following config keys are used to create the dataset:
        input: The input type should be "dataset".
        input_config: A dict containing the following key and values:
            `format`: str, speciifies the format of the input data. This will be the
            format that ray dataset supports. See ray.data.Dataset for
            supported formats. Only "parquet" or "json" are supported for now.
            `paths`: str, a single string or a list of strings. Each string is a path
            to a file or a directory holding the dataset. It can be either a local path
            or a remote path (e.g. to an s3 bucket).
            `loader_fn`: Callable[None, ray.data.Dataset], Instead of
            specifying paths and format, you can specify a function to load the dataset.
            `parallelism`: int, The number of tasks to use for loading the dataset.
            If not specified, it will be set to the number of workers.
            `num_cpus_per_read_task`: float, The number of CPUs to use for each read
            task. If not specified, it will be set to 0.5.

    Args:
        config: The config dict for the algorithm.
        num_workers: The number of shards to create for remote workers.

    Returns:
        dataset: The dataset object.
        shards: A list of dataset shards. For num_workers > 0 the first returned
        shared would be a dummy None shard for local_worker.
    """
    assert config.input_ == 'dataset', f"Must specify config.input_ as 'dataset' if calling `get_dataset_and_shards`. Got {config.input_}"
    input_config = config.input_config
    format = input_config.get('format')
    supported_fmts = ['json', 'parquet']
    if format is not None and format not in supported_fmts:
        raise ValueError(f'Unsupported format {format}. Supported formats are {supported_fmts}')
    paths = input_config.get('paths')
    loader_fn = input_config.get('loader_fn')
    if loader_fn and (format or paths):
        raise ValueError('When using a `loader_fn`, you cannot specify a `format` or `path`.')
    if not (format and paths) and (not loader_fn):
        raise ValueError('Must specify either a `loader_fn` or a `format` and `path` in `input_config`.')
    if paths is not None:
        if isinstance(paths, str):
            paths = [paths]
        elif isinstance(paths, list):
            assert isinstance(paths[0], str), 'Paths must be a list of path strings.'
        else:
            raise ValueError('Paths must be a path string or a list of path strings.')
        paths = _unzip_if_needed(paths, format)
    parallelism = input_config.get('parallelism', num_workers or 1)
    cpus_per_task = input_config.get('num_cpus_per_read_task', DEFAULT_NUM_CPUS_PER_TASK)
    if loader_fn:
        dataset = loader_fn()
    elif format == 'json':
        dataset = ray.data.read_json(paths, parallelism=parallelism, ray_remote_args={'num_cpus': cpus_per_task})
    elif format == 'parquet':
        dataset = ray.data.read_parquet(paths, parallelism=parallelism, ray_remote_args={'num_cpus': cpus_per_task})
    else:
        raise ValueError('Un-supported Ray dataset format: ', format)
    if num_workers == 0:
        return (dataset, [dataset])
    else:
        remote_shards = dataset.repartition(num_blocks=num_workers, shuffle=False).split(num_workers)
        return (dataset, [None] + remote_shards)
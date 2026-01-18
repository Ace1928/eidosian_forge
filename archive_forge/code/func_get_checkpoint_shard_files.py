import json
import os
import re
import shutil
import sys
import tempfile
import traceback
import warnings
from concurrent import futures
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
from uuid import uuid4
import huggingface_hub
import requests
from huggingface_hub import (
from huggingface_hub.file_download import REGEX_COMMIT_HASH, http_get
from huggingface_hub.utils import (
from huggingface_hub.utils._deprecation import _deprecate_method
from requests.exceptions import HTTPError
from . import __version__, logging
from .generic import working_or_temp_dir
from .import_utils import (
from .logging import tqdm
def get_checkpoint_shard_files(pretrained_model_name_or_path, index_filename, cache_dir=None, force_download=False, proxies=None, resume_download=False, local_files_only=False, token=None, user_agent=None, revision=None, subfolder='', _commit_hash=None, **deprecated_kwargs):
    """
    For a given model:

    - download and cache all the shards of a sharded checkpoint if `pretrained_model_name_or_path` is a model ID on the
      Hub
    - returns the list of paths to all the shards, as well as some metadata.

    For the description of each arg, see [`PreTrainedModel.from_pretrained`]. `index_filename` is the full path to the
    index (downloaded and cached if `pretrained_model_name_or_path` is a model ID on the Hub).
    """
    import json
    use_auth_token = deprecated_kwargs.pop('use_auth_token', None)
    if use_auth_token is not None:
        warnings.warn('The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.', FutureWarning)
        if token is not None:
            raise ValueError('`token` and `use_auth_token` are both specified. Please set only the argument `token`.')
        token = use_auth_token
    if not os.path.isfile(index_filename):
        raise ValueError(f"Can't find a checkpoint index ({index_filename}) in {pretrained_model_name_or_path}.")
    with open(index_filename, 'r') as f:
        index = json.loads(f.read())
    shard_filenames = sorted(set(index['weight_map'].values()))
    sharded_metadata = index['metadata']
    sharded_metadata['all_checkpoint_keys'] = list(index['weight_map'].keys())
    sharded_metadata['weight_map'] = index['weight_map'].copy()
    if os.path.isdir(pretrained_model_name_or_path):
        shard_filenames = [os.path.join(pretrained_model_name_or_path, subfolder, f) for f in shard_filenames]
        return (shard_filenames, sharded_metadata)
    cached_filenames = []
    last_shard = try_to_load_from_cache(pretrained_model_name_or_path, shard_filenames[-1], cache_dir=cache_dir, revision=_commit_hash)
    show_progress_bar = last_shard is None or force_download
    for shard_filename in tqdm(shard_filenames, desc='Downloading shards', disable=not show_progress_bar):
        try:
            cached_filename = cached_file(pretrained_model_name_or_path, shard_filename, cache_dir=cache_dir, force_download=force_download, proxies=proxies, resume_download=resume_download, local_files_only=local_files_only, token=token, user_agent=user_agent, revision=revision, subfolder=subfolder, _commit_hash=_commit_hash)
        except EntryNotFoundError:
            raise EnvironmentError(f'{pretrained_model_name_or_path} does not appear to have a file named {shard_filename} which is required according to the checkpoint index.')
        except HTTPError:
            raise EnvironmentError(f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load {shard_filename}. You should try again after checking your internet connection.")
        cached_filenames.append(cached_filename)
    return (cached_filenames, sharded_metadata)
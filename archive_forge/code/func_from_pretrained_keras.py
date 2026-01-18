import collections.abc as collections
import json
import os
import warnings
from pathlib import Path
from shutil import copytree
from typing import Any, Dict, List, Optional, Union
from huggingface_hub import ModelHubMixin, snapshot_download
from huggingface_hub.utils import (
from .constants import CONFIG_NAME
from .hf_api import HfApi
from .utils import SoftTemporaryDirectory, logging, validate_hf_hub_args
def from_pretrained_keras(*args, **kwargs) -> 'KerasModelHubMixin':
    """
    Instantiate a pretrained Keras model from a pre-trained model from the Hub.
    The model is expected to be in `SavedModel` format.

    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            Can be either:
                - A string, the `model id` of a pretrained model hosted inside a
                  model repo on huggingface.co. Valid model ids can be located
                  at the root-level, like `bert-base-uncased`, or namespaced
                  under a user or organization name, like
                  `dbmdz/bert-base-german-cased`.
                - You can add `revision` by appending `@` at the end of model_id
                  simply like this: `dbmdz/bert-base-german-cased@main` Revision
                  is the specific model version to use. It can be a branch name,
                  a tag name, or a commit id, since we use a git-based system
                  for storing models and other artifacts on huggingface.co, so
                  `revision` can be any identifier allowed by git.
                - A path to a `directory` containing model weights saved using
                  [`~transformers.PreTrainedModel.save_pretrained`], e.g.,
                  `./my_model_directory/`.
                - `None` if you are both providing the configuration and state
                  dictionary (resp. with keyword arguments `config` and
                  `state_dict`).
        force_download (`bool`, *optional*, defaults to `False`):
            Whether to force the (re-)download of the model weights and
            configuration files, overriding the cached versions if they exist.
        resume_download (`bool`, *optional*, defaults to `False`):
            Whether to delete incompletely received files. Will attempt to
            resume the download if such a file exists.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g.,
            `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The
            proxies are used on each request.
        token (`str` or `bool`, *optional*):
            The token to use as HTTP bearer authorization for remote files. If
            `True`, will use the token generated when running `transformers-cli
            login` (stored in `~/.huggingface`).
        cache_dir (`Union[str, os.PathLike]`, *optional*):
            Path to a directory in which a downloaded pretrained model
            configuration should be cached if the standard cache should not be
            used.
        local_files_only(`bool`, *optional*, defaults to `False`):
            Whether to only look at local files (i.e., do not try to download
            the model).
        model_kwargs (`Dict`, *optional*):
            model_kwargs will be passed to the model during initialization

    <Tip>

    Passing `token=True` is required when you want to use a private
    model.

    </Tip>
    """
    return KerasModelHubMixin.from_pretrained(*args, **kwargs)
import importlib
import json
import os
import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union
from ...configuration_utils import PretrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import TOKENIZER_CONFIG_FILE
from ...utils import (
from ..encoder_decoder import EncoderDecoderConfig
from .auto_factory import _LazyAutoMapping
from .configuration_auto import (
def get_tokenizer_config(pretrained_model_name_or_path: Union[str, os.PathLike], cache_dir: Optional[Union[str, os.PathLike]]=None, force_download: bool=False, resume_download: bool=False, proxies: Optional[Dict[str, str]]=None, token: Optional[Union[bool, str]]=None, revision: Optional[str]=None, local_files_only: bool=False, subfolder: str='', **kwargs):
    """
    Loads the tokenizer configuration from a pretrained model tokenizer configuration.

    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
              huggingface.co.
            - a path to a *directory* containing a configuration file saved using the
              [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.

        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download (`bool`, *optional*, defaults to `False`):
            Whether or not to delete incompletely received file. Attempts to resume the download if such a file exists.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        subfolder (`str`, *optional*, defaults to `""`):
            In case the tokenizer config is located inside a subfolder of the model repo on huggingface.co, you can
            specify the folder name here.

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Dict`: The configuration of the tokenizer.

    Examples:

    ```python
    # Download configuration from huggingface.co and cache.
    tokenizer_config = get_tokenizer_config("google-bert/bert-base-uncased")
    # This model does not have a tokenizer config so the result will be an empty dict.
    tokenizer_config = get_tokenizer_config("FacebookAI/xlm-roberta-base")

    # Save a pretrained tokenizer locally and you can reload its config
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
    tokenizer.save_pretrained("tokenizer-test")
    tokenizer_config = get_tokenizer_config("tokenizer-test")
    ```"""
    use_auth_token = kwargs.pop('use_auth_token', None)
    if use_auth_token is not None:
        warnings.warn('The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.', FutureWarning)
        if token is not None:
            raise ValueError('`token` and `use_auth_token` are both specified. Please set only the argument `token`.')
        token = use_auth_token
    commit_hash = kwargs.get('_commit_hash', None)
    resolved_config_file = cached_file(pretrained_model_name_or_path, TOKENIZER_CONFIG_FILE, cache_dir=cache_dir, force_download=force_download, resume_download=resume_download, proxies=proxies, token=token, revision=revision, local_files_only=local_files_only, subfolder=subfolder, _raise_exceptions_for_gated_repo=False, _raise_exceptions_for_missing_entries=False, _raise_exceptions_for_connection_errors=False, _commit_hash=commit_hash)
    if resolved_config_file is None:
        logger.info('Could not locate the tokenizer configuration file, will try to use the model config instead.')
        return {}
    commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
    with open(resolved_config_file, encoding='utf-8') as reader:
        result = json.load(reader)
    result['_commit_hash'] = commit_hash
    return result
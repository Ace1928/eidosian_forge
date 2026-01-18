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
@validate_hf_hub_args
def push_to_hub_keras(model, repo_id: str, *, config: Optional[dict]=None, commit_message: str='Push Keras model using huggingface_hub.', private: bool=False, api_endpoint: Optional[str]=None, token: Optional[str]=None, branch: Optional[str]=None, create_pr: Optional[bool]=None, allow_patterns: Optional[Union[List[str], str]]=None, ignore_patterns: Optional[Union[List[str], str]]=None, delete_patterns: Optional[Union[List[str], str]]=None, log_dir: Optional[str]=None, include_optimizer: bool=False, tags: Optional[Union[list, str]]=None, plot_model: bool=True, **model_save_kwargs):
    """
    Upload model checkpoint to the Hub.

    Use `allow_patterns` and `ignore_patterns` to precisely filter which files should be pushed to the hub. Use
    `delete_patterns` to delete existing remote files in the same commit. See [`upload_folder`] reference for more
    details.

    Args:
        model (`Keras.Model`):
            The [Keras model](`https://www.tensorflow.org/api_docs/python/tf/keras/Model`) you'd like to push to the
            Hub. The model must be compiled and built.
        repo_id (`str`):
                ID of the repository to push to (example: `"username/my-model"`).
        commit_message (`str`, *optional*, defaults to "Add Keras model"):
            Message to commit while pushing.
        private (`bool`, *optional*, defaults to `False`):
            Whether the repository created should be private.
        api_endpoint (`str`, *optional*):
            The API endpoint to use when pushing the model to the hub.
        token (`str`, *optional*):
            The token to use as HTTP bearer authorization for remote files. If
            not set, will use the token set when logging in with
            `huggingface-cli login` (stored in `~/.huggingface`).
        branch (`str`, *optional*):
            The git branch on which to push the model. This defaults to
            the default branch as specified in your repository, which
            defaults to `"main"`.
        create_pr (`boolean`, *optional*):
            Whether or not to create a Pull Request from `branch` with that commit.
            Defaults to `False`.
        config (`dict`, *optional*):
            Configuration object to be saved alongside the model weights.
        allow_patterns (`List[str]` or `str`, *optional*):
            If provided, only files matching at least one pattern are pushed.
        ignore_patterns (`List[str]` or `str`, *optional*):
            If provided, files matching any of the patterns are not pushed.
        delete_patterns (`List[str]` or `str`, *optional*):
            If provided, remote files matching any of the patterns will be deleted from the repo.
        log_dir (`str`, *optional*):
            TensorBoard logging directory to be pushed. The Hub automatically
            hosts and displays a TensorBoard instance if log files are included
            in the repository.
        include_optimizer (`bool`, *optional*, defaults to `False`):
            Whether or not to include optimizer during serialization.
        tags (Union[`list`, `str`], *optional*):
            List of tags that are related to model or string of a single tag. See example tags
            [here](https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1).
        plot_model (`bool`, *optional*, defaults to `True`):
            Setting this to `True` will plot the model and put it in the model
            card. Requires graphviz and pydot to be installed.
        model_save_kwargs(`dict`, *optional*):
            model_save_kwargs will be passed to
            [`tf.keras.models.save_model()`](https://www.tensorflow.org/api_docs/python/tf/keras/models/save_model).

    Returns:
        The url of the commit of your model in the given repository.
    """
    api = HfApi(endpoint=api_endpoint)
    repo_id = api.create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True).repo_id
    with SoftTemporaryDirectory() as tmp:
        saved_path = Path(tmp) / repo_id
        save_pretrained_keras(model, saved_path, config=config, include_optimizer=include_optimizer, tags=tags, plot_model=plot_model, **model_save_kwargs)
        if log_dir is not None:
            delete_patterns = [] if delete_patterns is None else [delete_patterns] if isinstance(delete_patterns, str) else delete_patterns
            delete_patterns.append('logs/*')
            copytree(log_dir, saved_path / 'logs')
        return api.upload_folder(repo_type='model', repo_id=repo_id, folder_path=saved_path, commit_message=commit_message, token=token, revision=branch, create_pr=create_pr, allow_patterns=allow_patterns, ignore_patterns=ignore_patterns, delete_patterns=delete_patterns)
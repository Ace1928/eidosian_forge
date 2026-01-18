import os
import re
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Type, Union
import requests
import yaml
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.hf_api import upload_file
from huggingface_hub.repocard_data import (
from huggingface_hub.utils import get_session, is_jinja_available, yaml_dump
from .constants import REPOCARD_NAME
from .utils import EntryNotFoundError, SoftTemporaryDirectory, logging, validate_hf_hub_args
@validate_hf_hub_args
def metadata_update(repo_id: str, metadata: Dict, *, repo_type: Optional[str]=None, overwrite: bool=False, token: Optional[str]=None, commit_message: Optional[str]=None, commit_description: Optional[str]=None, revision: Optional[str]=None, create_pr: bool=False, parent_commit: Optional[str]=None) -> str:
    """
    Updates the metadata in the README.md of a repository on the Hugging Face Hub.
    If the README.md file doesn't exist yet, a new one is created with metadata and an
    the default ModelCard or DatasetCard template. For `space` repo, an error is thrown
    as a Space cannot exist without a `README.md` file.

    Args:
        repo_id (`str`):
            The name of the repository.
        metadata (`dict`):
            A dictionary containing the metadata to be updated.
        repo_type (`str`, *optional*):
            Set to `"dataset"` or `"space"` if updating to a dataset or space,
            `None` or `"model"` if updating to a model. Default is `None`.
        overwrite (`bool`, *optional*, defaults to `False`):
            If set to `True` an existing field can be overwritten, otherwise
            attempting to overwrite an existing field will cause an error.
        token (`str`, *optional*):
            The Hugging Face authentication token.
        commit_message (`str`, *optional*):
            The summary / title / first line of the generated commit. Defaults to
            `f"Update metadata with huggingface_hub"`
        commit_description (`str` *optional*)
            The description of the generated commit
        revision (`str`, *optional*):
            The git revision to commit from. Defaults to the head of the
            `"main"` branch.
        create_pr (`boolean`, *optional*):
            Whether or not to create a Pull Request from `revision` with that commit.
            Defaults to `False`.
        parent_commit (`str`, *optional*):
            The OID / SHA of the parent commit, as a hexadecimal string. Shorthands (7 first characters) are also supported.
            If specified and `create_pr` is `False`, the commit will fail if `revision` does not point to `parent_commit`.
            If specified and `create_pr` is `True`, the pull request will be created from `parent_commit`.
            Specifying `parent_commit` ensures the repo has not changed before committing the changes, and can be
            especially useful if the repo is updated / committed to concurrently.
    Returns:
        `str`: URL of the commit which updated the card metadata.

    Example:
        ```python
        >>> from huggingface_hub import metadata_update
        >>> metadata = {'model-index': [{'name': 'RoBERTa fine-tuned on ReactionGIF',
        ...             'results': [{'dataset': {'name': 'ReactionGIF',
        ...                                      'type': 'julien-c/reactiongif'},
        ...                           'metrics': [{'name': 'Recall',
        ...                                        'type': 'recall',
        ...                                        'value': 0.7762102282047272}],
        ...                          'task': {'name': 'Text Classification',
        ...                                   'type': 'text-classification'}}]}]}
        >>> url = metadata_update("hf-internal-testing/reactiongif-roberta-card", metadata)

        ```
    """
    commit_message = commit_message if commit_message is not None else 'Update metadata with huggingface_hub'
    card_class: Type[RepoCard]
    if repo_type is None or repo_type == 'model':
        card_class = ModelCard
    elif repo_type == 'dataset':
        card_class = DatasetCard
    elif repo_type == 'space':
        card_class = RepoCard
    else:
        raise ValueError(f'Unknown repo_type: {repo_type}')
    try:
        card = card_class.load(repo_id, token=token, repo_type=repo_type)
    except EntryNotFoundError:
        if repo_type == 'space':
            raise ValueError("Cannot update metadata on a Space that doesn't contain a `README.md` file.")
        card = card_class.from_template(CardData())
    for key, value in metadata.items():
        if key == 'model-index':
            if 'name' not in value[0]:
                value[0]['name'] = getattr(card, 'model_name', repo_id)
            model_name, new_results = model_index_to_eval_results(value)
            if card.data.eval_results is None:
                card.data.eval_results = new_results
                card.data.model_name = model_name
            else:
                existing_results = card.data.eval_results
                for new_result in new_results:
                    result_found = False
                    for existing_result in existing_results:
                        if new_result.is_equal_except_value(existing_result):
                            if new_result != existing_result and (not overwrite):
                                raise ValueError(f"You passed a new value for the existing metric 'name: {new_result.metric_name}, type: {new_result.metric_type}'. Set `overwrite=True` to overwrite existing metrics.")
                            result_found = True
                            existing_result.metric_value = new_result.metric_value
                            if existing_result.verified is True:
                                existing_result.verify_token = new_result.verify_token
                    if not result_found:
                        card.data.eval_results.append(new_result)
        elif card.data.get(key) is not None and (not overwrite) and (card.data.get(key) != value):
            raise ValueError(f"You passed a new value for the existing meta data field '{key}'. Set `overwrite=True` to overwrite existing metadata.")
        else:
            card.data[key] = value
    return card.push_to_hub(repo_id, token=token, repo_type=repo_type, commit_message=commit_message, commit_description=commit_description, create_pr=create_pr, revision=revision, parent_commit=parent_commit)
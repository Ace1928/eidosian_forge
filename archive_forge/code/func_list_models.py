from __future__ import annotations
import inspect
import json
import re
import struct
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import wraps
from itertools import islice
from pathlib import Path
from typing import (
from urllib.parse import quote
import requests
from requests.exceptions import HTTPError
from tqdm.auto import tqdm as base_tqdm
from tqdm.contrib.concurrent import thread_map
from ._commit_api import (
from ._inference_endpoints import InferenceEndpoint, InferenceEndpointType
from ._multi_commits import (
from ._space_api import SpaceHardware, SpaceRuntime, SpaceStorage, SpaceVariable
from .community import (
from .constants import (
from .file_download import HfFileMetadata, get_hf_file_metadata, hf_hub_url
from .repocard_data import DatasetCardData, ModelCardData, SpaceCardData
from .utils import (  # noqa: F401 # imported for backward compatibility
from .utils import tqdm as hf_tqdm
from .utils._deprecation import _deprecate_arguments, _deprecate_method
from .utils._typing import CallableT
from .utils.endpoint_helpers import (
@validate_hf_hub_args
def list_models(self, *, filter: Union[ModelFilter, str, Iterable[str], None]=None, author: Optional[str]=None, library: Optional[Union[str, List[str]]]=None, language: Optional[Union[str, List[str]]]=None, model_name: Optional[str]=None, task: Optional[Union[str, List[str]]]=None, trained_dataset: Optional[Union[str, List[str]]]=None, tags: Optional[Union[str, List[str]]]=None, search: Optional[str]=None, emissions_thresholds: Optional[Tuple[float, float]]=None, sort: Union[Literal['last_modified'], str, None]=None, direction: Optional[Literal[-1]]=None, limit: Optional[int]=None, full: Optional[bool]=None, cardData: bool=False, fetch_config: bool=False, token: Optional[Union[bool, str]]=None, pipeline_tag: Optional[str]=None) -> Iterable[ModelInfo]:
    """
        List models hosted on the Huggingface Hub, given some filters.

        Args:
            filter ([`ModelFilter`] or `str` or `Iterable`, *optional*):
                A string or [`ModelFilter`] which can be used to identify models
                on the Hub.
            author (`str`, *optional*):
                A string which identify the author (user or organization) of the
                returned models
            library (`str` or `List`, *optional*):
                A string or list of strings of foundational libraries models were
                originally trained from, such as pytorch, tensorflow, or allennlp.
            language (`str` or `List`, *optional*):
                A string or list of strings of languages, both by name and country
                code, such as "en" or "English"
            model_name (`str`, *optional*):
                A string that contain complete or partial names for models on the
                Hub, such as "bert" or "bert-base-cased"
            task (`str` or `List`, *optional*):
                A string or list of strings of tasks models were designed for, such
                as: "fill-mask" or "automatic-speech-recognition"
            trained_dataset (`str` or `List`, *optional*):
                A string tag or a list of string tags of the trained dataset for a
                model on the Hub.
            tags (`str` or `List`, *optional*):
                A string tag or a list of tags to filter models on the Hub by, such
                as `text-generation` or `spacy`.
            search (`str`, *optional*):
                A string that will be contained in the returned model ids.
            emissions_thresholds (`Tuple`, *optional*):
                A tuple of two ints or floats representing a minimum and maximum
                carbon footprint to filter the resulting models with in grams.
            sort (`Literal["last_modified"]` or `str`, *optional*):
                The key with which to sort the resulting models. Possible values
                are the properties of the [`huggingface_hub.hf_api.ModelInfo`] class.
            direction (`Literal[-1]` or `int`, *optional*):
                Direction in which to sort. The value `-1` sorts by descending
                order while all other values sort by ascending order.
            limit (`int`, *optional*):
                The limit on the number of models fetched. Leaving this option
                to `None` fetches all models.
            full (`bool`, *optional*):
                Whether to fetch all model data, including the `last_modified`,
                the `sha`, the files and the `tags`. This is set to `True` by
                default when using a filter.
            cardData (`bool`, *optional*):
                Whether to grab the metadata for the model as well. Can contain
                useful information such as carbon emissions, metrics, and
                datasets trained on.
            fetch_config (`bool`, *optional*):
                Whether to fetch the model configs as well. This is not included
                in `full` due to its size.
            token (`bool` or `str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token).
                If `None` or `True` and machine is logged in (through `huggingface-cli login`
                or [`~huggingface_hub.login`]), token will be retrieved from the cache.
                If `False`, token is not sent in the request header.
            pipeline_tag (`str`, *optional*):
                A string pipeline tag to filter models on the Hub by, such as `summarization`


        Returns:
            `Iterable[ModelInfo]`: an iterable of [`huggingface_hub.hf_api.ModelInfo`] objects.

        Example usage with the `filter` argument:

        ```python
        >>> from huggingface_hub import HfApi

        >>> api = HfApi()

        >>> # List all models
        >>> api.list_models()

        >>> # List only the text classification models
        >>> api.list_models(filter="text-classification")

        >>> # List only models from the AllenNLP library
        >>> api.list_models(filter="allennlp")
        ```

        Example usage with the `search` argument:

        ```python
        >>> from huggingface_hub import HfApi

        >>> api = HfApi()

        >>> # List all models with "bert" in their name
        >>> api.list_models(search="bert")

        >>> # List all models with "bert" in their name made by google
        >>> api.list_models(search="bert", author="google")
        ```
        """
    if emissions_thresholds is not None and cardData is None:
        raise ValueError('`emissions_thresholds` were passed without setting `cardData=True`.')
    path = f'{self.endpoint}/api/models'
    headers = self._build_hf_headers(token=token)
    params = {}
    filter_list = []
    if filter is not None:
        if isinstance(filter, ModelFilter):
            params = self._unpack_model_filter(filter)
        else:
            params.update({'filter': filter})
        params.update({'full': True})
    if author:
        params.update({'author': author})
    if model_name:
        params.update({'search': model_name})
    if library:
        filter_list.extend([library] if isinstance(library, str) else library)
    if task:
        filter_list.extend([task] if isinstance(task, str) else task)
    if trained_dataset:
        if not isinstance(trained_dataset, (list, tuple)):
            trained_dataset = [trained_dataset]
        for dataset in trained_dataset:
            if not dataset.startswith('dataset:'):
                dataset = f'dataset:{dataset}'
            filter_list.append(dataset)
    if language:
        filter_list.extend([language] if isinstance(language, str) else language)
    if tags:
        filter_list.extend([tags] if isinstance(tags, str) else tags)
    if search:
        params.update({'search': search})
    if sort is not None:
        params.update({'sort': 'lastModified' if sort == 'last_modified' else sort})
    if direction is not None:
        params.update({'direction': direction})
    if limit is not None:
        params.update({'limit': limit})
    if full is not None:
        if full:
            params.update({'full': True})
        elif 'full' in params:
            del params['full']
    if fetch_config:
        params.update({'config': True})
    if cardData:
        params.update({'cardData': True})
    if pipeline_tag:
        params.update({'pipeline_tag': pipeline_tag})
    filter_value = params.get('filter', [])
    if filter_value:
        filter_list.extend([filter_value] if isinstance(filter_value, str) else list(filter_value))
    params.update({'filter': filter_list})
    items = paginate(path, params=params, headers=headers)
    if limit is not None:
        items = islice(items, limit)
    for item in items:
        if 'siblings' not in item:
            item['siblings'] = None
        model_info = ModelInfo(**item)
        if emissions_thresholds is None or _is_emission_within_treshold(model_info, *emissions_thresholds):
            yield model_info
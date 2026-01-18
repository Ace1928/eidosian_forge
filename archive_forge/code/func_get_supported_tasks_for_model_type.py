import importlib
import inspect
import itertools
import os
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import huggingface_hub
from packaging import version
from requests.exceptions import ConnectionError as RequestsConnectionError
from transformers import AutoConfig, PretrainedConfig, is_tf_available, is_torch_available
from transformers.utils import SAFE_WEIGHTS_NAME, TF2_WEIGHTS_NAME, WEIGHTS_NAME, logging
from ..utils import CONFIG_NAME
from ..utils.import_utils import is_onnx_available
@staticmethod
def get_supported_tasks_for_model_type(model_type: str, exporter: str, model_name: Optional[str]=None, library_name: Optional[str]=None) -> TaskNameToExportConfigDict:
    """
        Retrieves the `task -> exporter backend config constructors` map from the model type.

        Args:
            model_type (`str`):
                The model type to retrieve the supported tasks for.
            exporter (`str`):
                The name of the exporter.
            model_name (`Optional[str]`, defaults to `None`):
                The name attribute of the model object, only used for the exception message.
            library_name (`Optional[str]`, defaults to `None`):
                The library name of the model. Can be any of "transformers", "timm", "diffusers", "sentence_transformers".

        Returns:
            `TaskNameToExportConfigDict`: The dictionary mapping each task to a corresponding `ExportConfig`
            constructor.
        """
    if library_name is None:
        logger.warning('Not passing the argument `library_name` to `get_supported_tasks_for_model_type` is deprecated and the support will be removed in a future version of Optimum. Please specify a `library_name`. Defaulting to `"transformers`.')
        supported_model_type_for_library = {**TasksManager._DIFFUSERS_SUPPORTED_MODEL_TYPE, **TasksManager._TIMM_SUPPORTED_MODEL_TYPE, **TasksManager._SENTENCE_TRANSFORMERS_SUPPORTED_MODEL_TYPE, **TasksManager._SUPPORTED_MODEL_TYPE}
        library_name = 'transformers'
    else:
        supported_model_type_for_library = TasksManager._LIBRARY_TO_SUPPORTED_MODEL_TYPES[library_name]
    model_type = model_type.lower().replace('_', '-')
    model_type_and_model_name = f'{model_type} ({model_name})' if model_name else model_type
    default_model_type = None
    if library_name in TasksManager._MODEL_TYPE_FOR_DEFAULT_CONFIG:
        default_model_type = TasksManager._MODEL_TYPE_FOR_DEFAULT_CONFIG[library_name]
    if model_type not in supported_model_type_for_library:
        if default_model_type is not None:
            model_type = default_model_type
        else:
            raise KeyError(f'{model_type_and_model_name} is not supported yet for {library_name}. Only {list(supported_model_type_for_library.keys())} are supported for the library {library_name}. If you want to support {model_type} please propose a PR or open up an issue.')
    if exporter not in supported_model_type_for_library[model_type]:
        raise KeyError(f'{model_type_and_model_name} is not supported yet with the {exporter} backend. Only {list(supported_model_type_for_library[model_type].keys())} are supported. If you want to support {exporter} please propose a PR or open up an issue.')
    return supported_model_type_for_library[model_type][exporter]
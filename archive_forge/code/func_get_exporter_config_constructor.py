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
def get_exporter_config_constructor(exporter: str, model: Optional[Union['PreTrainedModel', 'TFPreTrainedModel']]=None, task: str='feature-extraction', model_type: Optional[str]=None, model_name: Optional[str]=None, exporter_config_kwargs: Optional[Dict[str, Any]]=None, library_name: Optional[str]=None) -> ExportConfigConstructor:
    """
        Gets the `ExportConfigConstructor` for a model (or alternatively for a model type) and task combination.

        Args:
            exporter (`str`):
                The exporter to use.
            model (`Optional[Union[PreTrainedModel, TFPreTrainedModel]]`, defaults to `None`):
                The instance of the model.
            task (`str`, defaults to `"feature-extraction"`):
                The task to retrieve the config for.
            model_type (`Optional[str]`, defaults to `None`):
                The model type to retrieve the config for.
            model_name (`Optional[str]`, defaults to `None`):
                The name attribute of the model object, only used for the exception message.
            exporter_config_kwargs (`Optional[Dict[str, Any]]`, defaults to `None`):
                Arguments that will be passed to the exporter config class when building the config constructor.
            library_name (`Optional[str]`, defaults to `None`):
                The library name of the model. Can be any of "transformers", "timm", "diffusers", "sentence_transformers".

        Returns:
            `ExportConfigConstructor`: The `ExportConfig` constructor for the requested backend.
        """
    if library_name is None:
        logger.warning('Passing the argument `library_name` to `get_supported_tasks_for_model_type` is required, but got library_name=None. Defaulting to `transformers`. An error will be raised in a future version of Optimum if `library_name` is not provided.')
        supported_model_type_for_library = {**TasksManager._DIFFUSERS_SUPPORTED_MODEL_TYPE, **TasksManager._TIMM_SUPPORTED_MODEL_TYPE, **TasksManager._SENTENCE_TRANSFORMERS_SUPPORTED_MODEL_TYPE, **TasksManager._SUPPORTED_MODEL_TYPE}
        library_name = 'transformers'
    else:
        supported_model_type_for_library = TasksManager._LIBRARY_TO_SUPPORTED_MODEL_TYPES[library_name]
    if model is None and model_type is None:
        raise ValueError('Either a model_type or model should be provided to retrieve the export config.')
    if model_type is None:
        if hasattr(model.config, 'export_model_type'):
            model_type = model.config.export_model_type
        else:
            model_type = getattr(model.config, 'model_type', None)
        if model_type is None:
            raise ValueError('Model type cannot be inferred. Please provide the model_type for the model!')
        model_type = model_type.replace('_', '-')
        model_name = getattr(model, 'name', model_name)
    model_tasks = TasksManager.get_supported_tasks_for_model_type(model_type, exporter, model_name=model_name, library_name=library_name)
    if task not in model_tasks:
        synonyms = TasksManager.synonyms_for_task(task)
        for synonym in synonyms:
            if synonym in model_tasks:
                task = synonym
                break
        if task not in model_tasks:
            raise ValueError(f"{model_type} doesn't support task {task} for the {exporter} backend. Supported tasks are: {', '.join(model_tasks.keys())}.")
    if model_type not in supported_model_type_for_library:
        model_type = TasksManager._MODEL_TYPE_FOR_DEFAULT_CONFIG[library_name]
    exporter_config_constructor = supported_model_type_for_library[model_type][exporter][task]
    if exporter_config_kwargs is not None:
        exporter_config_constructor = partial(exporter_config_constructor, **exporter_config_kwargs)
    return exporter_config_constructor
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
def get_model_class_for_task(task: str, framework: str='pt', model_type: Optional[str]=None, model_class_name: Optional[str]=None, library: str='transformers') -> Type:
    """
        Attempts to retrieve an AutoModel class from a task name.

        Args:
            task (`str`):
                The task required.
            framework (`str`, defaults to `"pt"`):
                The framework to use for the export.
            model_type (`Optional[str]`, defaults to `None`):
                The model type to retrieve the model class for. Some architectures need a custom class to be loaded,
                and can not be loaded from auto class.
            model_class_name (`Optional[str]`, defaults to `None`):
                A model class name, allowing to override the default class that would be detected for the task. This
                parameter is useful for example for "automatic-speech-recognition", that may map to
                AutoModelForSpeechSeq2Seq or to AutoModelForCTC.
            library (`str`, defaults to `transformers`):
                The library name of the model. Can be any of "transformers", "timm", "diffusers", "sentence_transformers".

        Returns:
            The AutoModel class corresponding to the task.
        """
    task = task.replace('-with-past', '')
    task = TasksManager.map_from_synonym(task)
    TasksManager._validate_framework_choice(framework)
    if (framework, model_type, task) in TasksManager._CUSTOM_CLASSES:
        library, class_name = TasksManager._CUSTOM_CLASSES[framework, model_type, task]
        loaded_library = importlib.import_module(library)
        return getattr(loaded_library, class_name)
    else:
        if framework == 'pt':
            tasks_to_model_loader = TasksManager._LIBRARY_TO_TASKS_TO_MODEL_LOADER_MAP[library]
        else:
            tasks_to_model_loader = TasksManager._LIBRARY_TO_TF_TASKS_TO_MODEL_LOADER_MAP[library]
        loaded_library = importlib.import_module(library)
        if model_class_name is None:
            if task not in tasks_to_model_loader:
                raise KeyError(f'Unknown task: {task}. Possible values are: ' + ', '.join([f'`{key}` for {tasks_to_model_loader[key]}' for key in tasks_to_model_loader]))
            if isinstance(tasks_to_model_loader[task], str):
                model_class_name = tasks_to_model_loader[task]
            elif library == 'transformers':
                if model_type is None:
                    logger.warning(f'No model type passed for the task {task}, that may be mapped to several loading classes ({tasks_to_model_loader[task]}). Defaulting to {tasks_to_model_loader[task][0]} to load the model.')
                    model_class_name = tasks_to_model_loader[task][0]
                else:
                    for autoclass_name in tasks_to_model_loader[task]:
                        module = getattr(loaded_library, autoclass_name)
                        if model_type in module._model_mapping._model_mapping or model_type.replace('-', '_') in module._model_mapping._model_mapping:
                            model_class_name = autoclass_name
                            break
                    if model_class_name is None:
                        raise ValueError(f'Unrecognized configuration classes {tasks_to_model_loader[task]} do not match with the model type {model_type} and task {task}.')
            else:
                raise NotImplementedError('For library other than transformers, the _TASKS_TO_MODEL_LOADER mapping should be one to one.')
        return getattr(loaded_library, model_class_name)
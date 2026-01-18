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
@classmethod
def infer_task_from_model(cls, model: Union[str, 'PreTrainedModel', 'TFPreTrainedModel', Type], subfolder: str='', revision: Optional[str]=None) -> str:
    """
        Infers the task from the model repo.

        Args:
            model (`str`):
                The model to infer the task from. This can either be the name of a repo on the HuggingFace Hub, an
                instance of a model, or a model class.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the model files are located inside a subfolder of the model directory / repo on the Hugging
                Face Hub, you can specify the subfolder name here.
            revision (`Optional[str]`,  defaults to `None`):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id.
        Returns:
            `str`: The task name automatically detected from the model repo.
        """
    is_torch_pretrained_model = is_torch_available() and isinstance(model, PreTrainedModel)
    is_tf_pretrained_model = is_tf_available() and isinstance(model, TFPreTrainedModel)
    task = None
    if isinstance(model, str):
        task = cls._infer_task_from_model_name_or_path(model, subfolder=subfolder, revision=revision)
    elif is_torch_pretrained_model or is_tf_pretrained_model:
        task = cls._infer_task_from_model_or_model_class(model=model)
    elif inspect.isclass(model):
        task = cls._infer_task_from_model_or_model_class(model_class=model)
    if task is None:
        raise ValueError(f'Could not infer the task from {model}.')
    return task
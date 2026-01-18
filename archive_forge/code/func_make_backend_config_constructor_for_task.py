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
def make_backend_config_constructor_for_task(config_cls: Type, task: str) -> ExportConfigConstructor:
    if '-with-past' in task:
        if not getattr(config_cls, 'SUPPORTS_PAST', False):
            raise ValueError(f'{config_cls} does not support tasks with past.')
        constructor = partial(config_cls, use_past=True, task=task.replace('-with-past', ''))
    else:
        constructor = partial(config_cls, task=task)
    return constructor
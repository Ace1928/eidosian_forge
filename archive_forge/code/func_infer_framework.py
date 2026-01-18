import inspect
import tempfile
from collections import OrderedDict, UserDict
from collections.abc import MutableMapping
from contextlib import ExitStack, contextmanager
from dataclasses import fields, is_dataclass
from enum import Enum
from functools import partial
from typing import Any, ContextManager, Iterable, List, Tuple
import numpy as np
from packaging import version
from .import_utils import get_torch_version, is_flax_available, is_tf_available, is_torch_available, is_torch_fx_proxy
def infer_framework(model_class):
    """
    Infers the framework of a given model without using isinstance(), because we cannot guarantee that the relevant
    classes are imported or available.
    """
    for base_class in inspect.getmro(model_class):
        module = base_class.__module__
        name = base_class.__name__
        if module.startswith('tensorflow') or module.startswith('keras') or name == 'TFPreTrainedModel':
            return 'tf'
        elif module.startswith('torch') or name == 'PreTrainedModel':
            return 'pt'
        elif module.startswith('flax') or module.startswith('jax') or name == 'FlaxPreTrainedModel':
            return 'flax'
    else:
        raise TypeError(f'Could not infer framework from class {model_class}.')
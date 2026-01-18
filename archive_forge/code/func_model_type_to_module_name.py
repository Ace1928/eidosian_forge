import importlib
import os
import re
import warnings
from collections import OrderedDict
from typing import List, Union
from ...configuration_utils import PretrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...utils import CONFIG_NAME, logging
def model_type_to_module_name(key):
    """Converts a config key to the corresponding module."""
    if key in SPECIAL_MODEL_TYPE_TO_MODULE_NAME:
        return SPECIAL_MODEL_TYPE_TO_MODULE_NAME[key]
    key = key.replace('-', '_')
    if key in DEPRECATED_MODELS:
        key = f'deprecated.{key}'
    return key
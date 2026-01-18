import importlib
import os
import re
import warnings
from collections import OrderedDict
from typing import List, Union
from ...configuration_utils import PretrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...utils import CONFIG_NAME, logging
def config_class_to_model_type(config):
    """Converts a config class name to the corresponding model type"""
    for key, cls in CONFIG_MAPPING_NAMES.items():
        if cls == config:
            return key
    for key, cls in CONFIG_MAPPING._extra_content.items():
        if cls.__name__ == config:
            return key
    return None
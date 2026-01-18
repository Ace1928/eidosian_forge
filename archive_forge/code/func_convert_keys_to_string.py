import copy
import json
import os
import warnings
from typing import Any, Dict, Optional, Union
from .. import __version__
from ..configuration_utils import PretrainedConfig
from ..utils import (
def convert_keys_to_string(obj):
    if isinstance(obj, dict):
        return {str(key): convert_keys_to_string(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_string(item) for item in obj]
    else:
        return obj
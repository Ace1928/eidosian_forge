import dataclasses
import functools
import inspect
import math
import sys
import types
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import transformers
from packaging import version
from transformers.models.speecht5.modeling_speecht5 import SpeechT5EncoderWithSpeechPrenet
from transformers.utils import is_torch_available
from ...configuration_utils import _transformers_version
from ...utils import logging
def patch_everywhere(attribute_name: str, patch: Any, module_name_prefix: Optional[str]=None):
    """
    Finds all occurences of `attribute_name` in the loaded modules and patches them with `patch`.

    Args:
        attribute_name (`str`):
            The name of attribute to patch.
        patch (`Any`):
            The patch for the attribute.
        module_name_prefix (`Optional[str]`, defaults to `None`):
            If set, only module names starting with this prefix will be considered for patching.
    """
    for name in list(sys.modules):
        module = sys.modules[name]
        if module_name_prefix is not None and (not name.startswith(module_name_prefix)):
            continue
        if hasattr(module, attribute_name):
            setattr(module, attribute_name, patch)
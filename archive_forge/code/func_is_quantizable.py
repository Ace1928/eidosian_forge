import copy
import importlib.metadata
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from packaging import version
from ..utils import is_auto_awq_available, is_torch_available, logging
def is_quantizable(self):
    """
        Returns `True` if the model is quantizable, `False` otherwise.
        """
    return self.load_in_8bit or self.load_in_4bit
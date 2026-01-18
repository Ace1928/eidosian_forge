import copy
import importlib.metadata
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from packaging import version
from ..utils import is_auto_awq_available, is_torch_available, logging
@load_in_4bit.setter
def load_in_4bit(self, value: bool):
    if self.load_in_8bit and value:
        raise ValueError('load_in_4bit and load_in_8bit are both True, but only one can be used at the same time')
    self._load_in_4bit = value
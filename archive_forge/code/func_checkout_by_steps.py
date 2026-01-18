import copy
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union
import torch
from torch import Tensor
from .pygit import PyGit
from .sha1_store import SHA1_Store
def checkout_by_steps(self) -> None:
    """Not Implemented: Checkout by step count of the train process"""
    self._sanity_check()
    raise NotImplementedError()
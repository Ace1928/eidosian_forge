import inspect
import re
import string
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
def normalize_inputs(x: InputsType) -> ExportArgs:
    if isinstance(x, tuple):
        return ExportArgs(*x)
    assert isinstance(x, ExportArgs)
    return x
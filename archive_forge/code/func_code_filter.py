import inspect
import pathlib
import sys
import typing
from collections import defaultdict
from types import CodeType
from typing import Dict, Iterable, List, Optional
import torch
def code_filter(self) -> Optional[CodeFilter]:
    return jit_code_filter
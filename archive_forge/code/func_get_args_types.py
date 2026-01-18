import inspect
import pathlib
import sys
import typing
from collections import defaultdict
from types import CodeType
from typing import Dict, Iterable, List, Optional
import torch
def get_args_types(self, qualified_name: str) -> Dict:
    return self.consolidate_types(qualified_name)
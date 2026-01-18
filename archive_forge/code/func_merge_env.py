import contextlib
import functools
import hashlib
import os
import re
import sys
import textwrap
from argparse import Namespace
from dataclasses import fields, is_dataclass
from enum import auto, Enum
from typing import (
from typing_extensions import Self
from torchgen.code_template import CodeTemplate
def merge_env(into: Dict[str, List[str]], from_: Dict[str, List[str]]) -> None:
    for k, v in from_.items():
        assert k in sharded_keys, f'undeclared sharded key {k}'
        into[k] += v
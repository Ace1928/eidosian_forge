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
def dataclass_repr(obj: Any, indent: int=0, width: int=80) -> str:
    if sys.version_info >= (3, 10):
        from pprint import pformat
        return pformat(obj, indent, width)
    return _pformat(obj, indent=indent, width=width)
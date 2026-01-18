import collections.abc
import dataclasses
import enum
import inspect
import os
import pathlib
from collections import deque
from typing import (
from typing_extensions import Annotated, Final, Literal, get_args, get_origin
from . import _resolver
from . import _strings
from ._typing import TypeForm
from .conf import _markers
def append_sequence_instantiator(strings: List[List[str]]) -> Any:
    assert strings is not None
    return container_type((cast(_StandardInstantiator, make)(s) for s in strings))
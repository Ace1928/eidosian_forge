from __future__ import annotations
import dataclasses
import enum
import functools
import inspect
from inspect import Parameter
from inspect import signature
import os
from pathlib import Path
import sys
from typing import Any
from typing import Callable
from typing import Final
from typing import NoReturn
import py
def get_default_arg_names(function: Callable[..., Any]) -> tuple[str, ...]:
    return tuple((p.name for p in signature(function).parameters.values() if p.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY) and p.default is not Parameter.empty))
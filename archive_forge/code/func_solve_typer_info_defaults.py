import inspect
import os
import sys
import traceback
from datetime import datetime
from enum import Enum
from functools import update_wrapper
from pathlib import Path
from traceback import FrameSummary, StackSummary
from types import TracebackType
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
from uuid import UUID
import click
from .completion import get_completion_inspect_parameters
from .core import MarkupMode, TyperArgument, TyperCommand, TyperGroup, TyperOption
from .models import (
from .utils import get_params_from_function
def solve_typer_info_defaults(typer_info: TyperInfo) -> TyperInfo:
    values: Dict[str, Any] = {}
    name = None
    for name, value in typer_info.__dict__.items():
        if not isinstance(value, DefaultPlaceholder):
            values[name] = value
            continue
        try:
            callback_value = getattr(typer_info.typer_instance.registered_callback, name)
            if not isinstance(callback_value, DefaultPlaceholder):
                values[name] = callback_value
                continue
        except AttributeError:
            pass
        try:
            instance_value = getattr(typer_info.typer_instance.info, name)
            if not isinstance(instance_value, DefaultPlaceholder):
                values[name] = instance_value
                continue
        except AttributeError:
            pass
        values[name] = value.value
    if values['name'] is None:
        values['name'] = get_group_name(typer_info)
    values['help'] = solve_typer_info_help(typer_info)
    return TyperInfo(**values)
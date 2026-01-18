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
def get_click_type(*, annotation: Any, parameter_info: ParameterInfo) -> click.ParamType:
    if parameter_info.click_type is not None:
        return parameter_info.click_type
    elif parameter_info.parser is not None:
        return click.types.FuncParamType(parameter_info.parser)
    elif annotation == str:
        return click.STRING
    elif annotation == int:
        if parameter_info.min is not None or parameter_info.max is not None:
            min_ = None
            max_ = None
            if parameter_info.min is not None:
                min_ = int(parameter_info.min)
            if parameter_info.max is not None:
                max_ = int(parameter_info.max)
            return click.IntRange(min=min_, max=max_, clamp=parameter_info.clamp)
        else:
            return click.INT
    elif annotation == float:
        if parameter_info.min is not None or parameter_info.max is not None:
            return click.FloatRange(min=parameter_info.min, max=parameter_info.max, clamp=parameter_info.clamp)
        else:
            return click.FLOAT
    elif annotation == bool:
        return click.BOOL
    elif annotation == UUID:
        return click.UUID
    elif annotation == datetime:
        return click.DateTime(formats=parameter_info.formats)
    elif annotation == Path or parameter_info.allow_dash or parameter_info.path_type or parameter_info.resolve_path:
        return click.Path(exists=parameter_info.exists, file_okay=parameter_info.file_okay, dir_okay=parameter_info.dir_okay, writable=parameter_info.writable, readable=parameter_info.readable, resolve_path=parameter_info.resolve_path, allow_dash=parameter_info.allow_dash, path_type=parameter_info.path_type)
    elif lenient_issubclass(annotation, FileTextWrite):
        return click.File(mode=parameter_info.mode or 'w', encoding=parameter_info.encoding, errors=parameter_info.errors, lazy=parameter_info.lazy, atomic=parameter_info.atomic)
    elif lenient_issubclass(annotation, FileText):
        return click.File(mode=parameter_info.mode or 'r', encoding=parameter_info.encoding, errors=parameter_info.errors, lazy=parameter_info.lazy, atomic=parameter_info.atomic)
    elif lenient_issubclass(annotation, FileBinaryRead):
        return click.File(mode=parameter_info.mode or 'rb', encoding=parameter_info.encoding, errors=parameter_info.errors, lazy=parameter_info.lazy, atomic=parameter_info.atomic)
    elif lenient_issubclass(annotation, FileBinaryWrite):
        return click.File(mode=parameter_info.mode or 'wb', encoding=parameter_info.encoding, errors=parameter_info.errors, lazy=parameter_info.lazy, atomic=parameter_info.atomic)
    elif lenient_issubclass(annotation, Enum):
        return click.Choice([item.value for item in annotation], case_sensitive=parameter_info.case_sensitive)
    raise RuntimeError(f'Type not yet supported: {annotation}')
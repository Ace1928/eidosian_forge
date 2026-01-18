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
def generate_tuple_convertor(types: Sequence[Any]) -> Callable[[Optional[Tuple[Any, ...]]], Optional[Tuple[Any, ...]]]:
    convertors = [determine_type_convertor(type_) for type_ in types]

    def internal_convertor(param_args: Optional[Tuple[Any, ...]]) -> Optional[Tuple[Any, ...]]:
        if param_args is None:
            return None
        return tuple((convertor(arg) if convertor else arg for convertor, arg in zip(convertors, param_args)))
    return internal_convertor
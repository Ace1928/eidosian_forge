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
def add_typer(self, typer_instance: 'Typer', *, name: Optional[str]=Default(None), cls: Optional[Type[TyperGroup]]=Default(None), invoke_without_command: bool=Default(False), no_args_is_help: bool=Default(False), subcommand_metavar: Optional[str]=Default(None), chain: bool=Default(False), result_callback: Optional[Callable[..., Any]]=Default(None), context_settings: Optional[Dict[Any, Any]]=Default(None), callback: Optional[Callable[..., Any]]=Default(None), help: Optional[str]=Default(None), epilog: Optional[str]=Default(None), short_help: Optional[str]=Default(None), options_metavar: str=Default('[OPTIONS]'), add_help_option: bool=Default(True), hidden: bool=Default(False), deprecated: bool=Default(False), rich_help_panel: Union[str, None]=Default(None)) -> None:
    self.registered_groups.append(TyperInfo(typer_instance, name=name, cls=cls, invoke_without_command=invoke_without_command, no_args_is_help=no_args_is_help, subcommand_metavar=subcommand_metavar, chain=chain, result_callback=result_callback, context_settings=context_settings, callback=callback, help=help, epilog=epilog, short_help=short_help, options_metavar=options_metavar, add_help_option=add_help_option, hidden=hidden, deprecated=deprecated, rich_help_panel=rich_help_panel))
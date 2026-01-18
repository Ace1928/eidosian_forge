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
def get_click_param(param: ParamMeta) -> Tuple[Union[click.Argument, click.Option], Any]:
    default_value = None
    required = False
    if isinstance(param.default, ParameterInfo):
        parameter_info = param.default
        if parameter_info.default == Required:
            required = True
        else:
            default_value = parameter_info.default
    elif param.default == Required or param.default == param.empty:
        required = True
        parameter_info = ArgumentInfo()
    else:
        default_value = param.default
        parameter_info = OptionInfo()
    annotation: Any = Any
    if not param.annotation == param.empty:
        annotation = param.annotation
    else:
        annotation = str
    main_type = annotation
    is_list = False
    is_tuple = False
    parameter_type: Any = None
    is_flag = None
    origin = getattr(main_type, '__origin__', None)
    if origin is not None:
        if origin is Union:
            types = []
            for type_ in main_type.__args__:
                if type_ is NoneType:
                    continue
                types.append(type_)
            assert len(types) == 1, "Typer Currently doesn't support Union types"
            main_type = types[0]
            origin = getattr(main_type, '__origin__', None)
        if lenient_issubclass(origin, List):
            main_type = main_type.__args__[0]
            assert not getattr(main_type, '__origin__', None), 'List types with complex sub-types are not currently supported'
            is_list = True
        elif lenient_issubclass(origin, Tuple):
            types = []
            for type_ in main_type.__args__:
                assert not getattr(type_, '__origin__', None), 'Tuple types with complex sub-types are not currently supported'
                types.append(get_click_type(annotation=type_, parameter_info=parameter_info))
            parameter_type = tuple(types)
            is_tuple = True
    if parameter_type is None:
        parameter_type = get_click_type(annotation=main_type, parameter_info=parameter_info)
    convertor = determine_type_convertor(main_type)
    if is_list:
        convertor = generate_list_convertor(convertor=convertor, default_value=default_value)
    if is_tuple:
        convertor = generate_tuple_convertor(main_type.__args__)
    if isinstance(parameter_info, OptionInfo):
        if main_type is bool and parameter_info.is_flag is not False:
            is_flag = True
            parameter_type = None
        default_option_name = get_command_name(param.name)
        if is_flag:
            default_option_declaration = f'--{default_option_name}/--no-{default_option_name}'
        else:
            default_option_declaration = f'--{default_option_name}'
        param_decls = [param.name]
        if parameter_info.param_decls:
            param_decls.extend(parameter_info.param_decls)
        else:
            param_decls.append(default_option_declaration)
        return (TyperOption(param_decls=param_decls, show_default=parameter_info.show_default, prompt=parameter_info.prompt, confirmation_prompt=parameter_info.confirmation_prompt, prompt_required=parameter_info.prompt_required, hide_input=parameter_info.hide_input, is_flag=is_flag, flag_value=parameter_info.flag_value, multiple=is_list, count=parameter_info.count, allow_from_autoenv=parameter_info.allow_from_autoenv, type=parameter_type, help=parameter_info.help, hidden=parameter_info.hidden, show_choices=parameter_info.show_choices, show_envvar=parameter_info.show_envvar, required=required, default=default_value, callback=get_param_callback(callback=parameter_info.callback, convertor=convertor), metavar=parameter_info.metavar, expose_value=parameter_info.expose_value, is_eager=parameter_info.is_eager, envvar=parameter_info.envvar, shell_complete=parameter_info.shell_complete, autocompletion=get_param_completion(parameter_info.autocompletion), rich_help_panel=parameter_info.rich_help_panel), convertor)
    elif isinstance(parameter_info, ArgumentInfo):
        param_decls = [param.name]
        nargs = None
        if is_list:
            nargs = -1
        return (TyperArgument(param_decls=param_decls, type=parameter_type, required=required, nargs=nargs, show_default=parameter_info.show_default, show_choices=parameter_info.show_choices, show_envvar=parameter_info.show_envvar, help=parameter_info.help, hidden=parameter_info.hidden, default=default_value, callback=get_param_callback(callback=parameter_info.callback, convertor=convertor), metavar=parameter_info.metavar, expose_value=parameter_info.expose_value, is_eager=parameter_info.is_eager, envvar=parameter_info.envvar, autocompletion=get_param_completion(parameter_info.autocompletion), rich_help_panel=parameter_info.rich_help_panel), convertor)
    raise AssertionError('A click.Parameter should be returned')
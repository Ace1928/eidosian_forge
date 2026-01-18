from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
def dispatch(func, lib_name: str, lib_path: str, args: list, arg_type_symbol_dict: dict, ret_shape: tuple, is_pure: bool, _builder=None):
    """
        Dispatch a function to a library
        :param func: the function to dispatch
        :param lib_name: the name of the library
        :param lib_path: the path of the library
        :param args: the arguments of the function
        :param arg_type_symbol_dict: the type of the arguments
        :param ret_shape: the shape of the return value
        :param _builder: the builder
        :return: the return value of the function
    """
    if len(arg_type_symbol_dict) == 0:
        raise ValueError('arg_type_symbol_dict is empty')
    num_args = len(list(arg_type_symbol_dict.keys())[0])
    if len(args) != num_args:
        raise ValueError(f'length of input args does not match.Expect {len(args)}, got {num_args}')
    arg_types = []
    arg_list = []
    for arg in args:
        if isinstance(arg, tensor):
            arg_types.append(arg.dtype)
            arg_list.append(arg.handle)
        else:
            arg_types.append(type(arg))
            arg_list.append(arg)
    arg_types = tuple(arg_types)
    if arg_types not in arg_type_symbol_dict:
        raise ValueError(f'input arg type does not match.Expect one of {arg_type_symbol_dict.keys()}, got {arg_types}')
    else:
        symbol = arg_type_symbol_dict[arg_types][0]
        ret_type = arg_type_symbol_dict[arg_types][1]
        if ret_shape:
            ret_type = block_type(ret_type, ret_shape)
        return tensor(func(lib_name, lib_path, symbol, arg_list, ret_type.to_ir(_builder), is_pure), ret_type)
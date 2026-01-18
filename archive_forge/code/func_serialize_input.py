import base64
import dataclasses
import io
import json
import logging
import math
import operator
import typing
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Union
import sympy
import torch
import torch.export.exported_program as ep
from torch._export.verifier import load_verifier
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental import symbolic_shapes
from torch.utils._pytree import treespec_dumps, treespec_loads
from torch.utils._sympy.value_ranges import ValueRanges
from .schema import (  # type: ignore[attr-defined]
from torch.export.exported_program import (
from .upgrade import GraphModuleOpUpgrader
def serialize_input(self, arg) -> Argument:
    import torch._inductor.ir as inductor_ir
    inductor_tensor_buffers = (inductor_ir.Buffer, inductor_ir.ReinterpretView)
    if isinstance(arg, torch.fx.Node):
        if arg.op == 'get_attr':
            assert isinstance(arg.target, str)
            attr = getattr(arg.graph.owning_module, arg.target)
            if isinstance(attr, torch.Tensor):
                raise SerializeError('getattr nodes containing tensors should not appear in the graph')
            elif isinstance(attr, torch.fx.GraphModule):
                with self.save_graph_state():
                    graph = self.serialize_graph(attr)
                return Argument.create(as_graph=GraphArgument(name=arg.target, graph=graph))
            else:
                raise SerializeError(f'Unsupported getattr attribute {arg.target} with type: {type(attr)}')
        elif self.is_sym_int_arg(arg):
            return Argument.create(as_sym_int=SymIntArgument.create(as_name=arg.name))
        elif self.is_sym_bool_arg(arg):
            return Argument.create(as_sym_bool=SymBoolArgument.create(as_name=arg.name))
        else:
            return Argument.create(as_tensor=TensorArgument(name=arg.name))
    elif isinstance(arg, inductor_tensor_buffers):
        arg_name = arg.get_name()
        assert arg_name is not None, 'Buffer must have valid name'
        return Argument.create(as_tensor=TensorArgument(name=arg_name))
    elif isinstance(arg, torch.SymInt):
        return Argument.create(as_sym_int=SymIntArgument.create(as_name=str(arg)))
    elif isinstance(arg, bool):
        return Argument.create(as_bool=arg)
    elif isinstance(arg, str):
        return Argument.create(as_string=arg)
    elif isinstance(arg, int):
        return Argument.create(as_int=arg)
    elif isinstance(arg, float):
        return Argument.create(as_float=arg)
    elif arg is None:
        return Argument.create(as_none=())
    elif isinstance(arg, (list, tuple)):
        if all((isinstance(a, bool) for a in arg)):
            return Argument.create(as_bools=list(arg))
        elif all((isinstance(a, int) for a in arg)):
            return Argument.create(as_ints=list(arg))
        elif all((isinstance(a, float) for a in arg)):
            return Argument.create(as_floats=list(arg))
        elif all((isinstance(a, str) for a in arg)):
            return Argument.create(as_strings=list(arg))
        elif all((isinstance(a, torch.SymInt) for a in arg)):
            return Argument.create(as_sym_ints=[SymIntArgument.create(as_name=str(a)) for a in arg])
        elif all((self.is_sym_int_arg(a) for a in arg)):
            values = []
            for a in arg:
                if isinstance(a, torch.fx.Node):
                    values.append(SymIntArgument.create(as_name=a.name))
                elif isinstance(a, int):
                    values.append(SymIntArgument.create(as_int=a))
            return Argument.create(as_sym_ints=values)
        elif all((self.is_sym_bool_arg(a) for a in arg)):
            values = []
            for a in arg:
                if isinstance(a, torch.fx.Node):
                    values.append(SymBoolArgument.create(as_name=a.name))
                elif isinstance(a, bool):
                    values.append(SymBoolArgument.create(as_bool=a))
            return Argument.create(as_sym_bools=values)
        elif all((isinstance(a, torch.fx.Node) for a in arg)):
            arguments = []
            for a in arg:
                if a.op == 'get_attr':
                    raise SerializeError('getattr nodes containing tensors should not appear in the graph')
                arguments.append(TensorArgument(name=a.name))
            return Argument.create(as_tensors=arguments)
        elif all((isinstance(a, (torch.fx.Node, type(None))) for a in arg)):

            def serialize_optional_tensor_args(a):
                if a is None:
                    return OptionalTensorArgument.create(as_none=())
                elif isinstance(a, torch.fx.Node):
                    return OptionalTensorArgument.create(as_tensor=a.name)
                else:
                    raise SerializeError(f'Unsupported list/tuple argument: {a}')
            return Argument.create(as_optional_tensors=list(map(serialize_optional_tensor_args, arg)))
        elif all((isinstance(a, inductor_tensor_buffers) for a in arg)):
            return Argument.create(as_tensors=[TensorArgument(name=a.get_name()) for a in arg])
        elif all((isinstance(a, (*inductor_tensor_buffers, type(None))) for a in arg)):

            def serialize_optional_tensor_args(a):
                if a is None:
                    return OptionalTensorArgument.create(as_none=())
                elif isinstance(a, inductor_tensor_buffers):
                    return OptionalTensorArgument.create(as_tensor=a.get_name())
                else:
                    raise SerializeError(f'Unsupported list/tuple argument: {a}')
            return Argument.create(as_optional_tensors=list(map(serialize_optional_tensor_args, arg)))
        else:
            raise SerializeError(f'Unsupported list/tuple argument type: {[type(a) for a in arg]}')
    elif isinstance(arg, torch.dtype):
        return Argument.create(as_scalar_type=_TORCH_TO_SERIALIZE_DTYPE[arg])
    elif isinstance(arg, torch.device):
        return Argument.create(as_device=Device(type=arg.type, index=arg.index))
    elif isinstance(arg, torch.memory_format):
        return Argument.create(as_memory_format=_TORCH_TO_SERIALIZE_MEMORY_FORMAT[arg])
    elif isinstance(arg, torch.layout):
        return Argument.create(as_layout=_TORCH_TO_SERIALIZE_LAYOUT[arg])
    elif isinstance(arg, torch._C.ScriptObject):
        if not (arg._has_method('__getstate__') and arg._has_method('__setstate__')):
            raise SerializeError(f'Unable to serialize custom class {arg}. Please define serialization methods via def_pickle().')
        custom_obj_name = f'_custom_obj_{len(self.custom_objs)}'
        self.custom_objs[custom_obj_name] = arg
        return Argument.create(as_custom_obj=CustomObjArgument(custom_obj_name))
    else:
        raise SerializeError(f'Unsupported argument type: {type(arg)}')
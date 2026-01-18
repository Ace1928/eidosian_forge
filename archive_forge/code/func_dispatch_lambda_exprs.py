from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
from torchgen.api import cpp
from torchgen.api.types import Binding, CppSignature, CppSignatureGroup
from torchgen.gen import pythonify_default
from torchgen.model import (
def dispatch_lambda_exprs(ps: PythonSignature, f: NativeFunction, *, symint: bool=True) -> DispatchLambdaArgumentExprs:
    arg_parser_outputs = arg_parser_output_exprs(ps, f, symint=symint)
    lambda_args = dispatch_lambda_args(ps, f, symint=symint)
    inits: List[str] = []
    lambda_args_exprs: Dict[str, str] = {}
    has_toptions = has_tensor_options(f)
    for a in ps.arguments(skip_tensor_options=True):
        name = a.name
        arg_parser_expr = arg_parser_outputs[a.name].expr
        if has_toptions and name == 'self':
            inits.extend([f'auto self = {arg_parser_expr};'])
            lambda_args_exprs[name] = name
        elif isinstance(a, PythonOutArgument) and len(a.outputs) > 1 and f.func.is_out_fn():
            inits.extend([f'auto out = {arg_parser_expr};'])
            for i, out_arg in enumerate(a.outputs):
                lambda_args_exprs[out_arg.name] = f'out[{i}]'
        elif str(a.type) == 'Dimname[]?':
            inits.extend([f'auto __{name} = {arg_parser_expr};', f'c10::optional<DimnameList> {name} = __{name} ? c10::make_optional(DimnameList(__{name}.value())) : c10::nullopt;'])
            lambda_args_exprs[name] = name
        else:
            lambda_args_exprs[name] = arg_parser_expr
    if ps.method:
        lambda_args_exprs['self'] = 'self'
    tensor_options_args_names = [a.name for a in ps.tensor_options_args]
    if has_toptions:
        if f.func.is_out_fn():
            raise RuntimeError(f'{f.func}: tensor options with output arg')
        for a in ps.tensor_options_args:
            if a.name not in TENSOR_OPTIONS_FIELDS:
                raise RuntimeError(f"{f.func}: unrecognized tensor options field '{a.name}' in python binding arguments")
            if str(a.type) != TENSOR_OPTIONS_FIELDS.get(a.name):
                raise RuntimeError(f"{f.func}: unrecognized type '{str(a.type)}' for tensor options field '{a.name}'")
        if not all((a in tensor_options_args_names for a in TENSOR_OPTIONS_FIELDS.keys())):
            raise RuntimeError(f'{f.func}: incomplete tensor options args: {tensor_options_args_names}')
        inits.append(f'const auto options = TensorOptions()\n    .dtype({arg_parser_outputs['dtype'].expr})\n    .device({arg_parser_outputs['device'].expr})\n    .layout({arg_parser_outputs['layout'].expr})\n    .requires_grad({arg_parser_outputs['requires_grad'].expr})\n    .pinned_memory({arg_parser_outputs['pin_memory'].expr});\ntorch::utils::maybe_initialize_cuda(options);\n')
        lambda_args_exprs['options'] = 'options'
    if not has_toptions and tensor_options_args_names:
        if 'dtype' in tensor_options_args_names:
            if not f.func.is_out_fn():
                raise RuntimeError(f'{f.func}: dtype in tensor_options_args without output arg')
            if not all((a in tensor_options_args_names for a in ('layout', 'device'))):
                raise RuntimeError(f'{f.func}: incomplete tensor options for output check')
            inits.append(f'check_out_type_matches({arg_parser_outputs['out'].expr}, {arg_parser_outputs['dtype'].expr},\n                       {arg_parser_outputs['dtype'].is_none_expr}, {arg_parser_outputs['layout'].expr},\n                       {arg_parser_outputs['device'].expr}, {arg_parser_outputs['device'].is_none_expr});\n')
        if 'requires_grad' not in tensor_options_args_names:
            raise RuntimeError(f'{f.func}: expected "requires_grad" in tensor_options_args absent, but found [{tensor_options_args_names}]')
    return DispatchLambdaArgumentExprs(exprs=tuple((lambda_args_exprs[a.name] for a in lambda_args)), inits=inits)
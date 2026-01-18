import contextlib
from typing import Optional, Union, List, Set, Dict, Any
import warnings
from dataclasses import dataclass
import torch
import torchgen
from torch._C import _len_torch_dispatch_stack, _get_dispatch_stack_at,\
def return_and_correct_aliasing(func, args, kwargs, out):
    """
    This function should be used by wrapper tensor ``__torch_dispatch__`` subclasses
    that would like to work with torch.compile. It ensures that the subclass
    properly implements the aliasing behavior of every op,
    which is needed for correctness in AOTAutograd.
    This function will handle:

        * When we see a view op, we will alias the storages of any
          input and output tensor subclasses

        * When we see an inplace or out= op, we will directly
          return the corresponding input tensor, instead of returning
          a (potentially) fresh output tensor.
    """
    schema_info = get_alias_info(func)

    def get_write_alias(x):
        if len(x.alias_set) == 0:
            return None
        alias_set = list(x.alias_set)
        assert len(alias_set) == 1
        if x.is_write:
            return alias_set[0]
        return None

    def get_arg_from_alias(output_alias, schema_info, args, kwargs):
        new_args, new_kwargs = torch.fx.operator_schemas.normalize_function(func, args=args, kwargs=kwargs)
        arg_indices = [i for i, a in enumerate(schema_info.args) if output_alias in a.alias_set]
        assert len(arg_indices) == 1
        idx = arg_indices[0]
        arg_info = schema_info.args[idx]
        if arg_info.name is not None and arg_info.name in new_kwargs:
            return new_kwargs[arg_info.name]
        return new_args[idx]
    _correct_storage_aliasing(func, schema_info, args, (out,) if not isinstance(out, tuple) else out)
    if torch.Tag.inplace_view in func.tags:
        mutated_args = [x for i, x in enumerate(args) if get_write_alias(schema_info.args[i]) is not None]
        assert len(mutated_args) == 1
        from torch._subclasses.functional_tensor import FunctionalTensor
        if not isinstance(mutated_args[0], FunctionalTensor):
            with torch.utils._mode_utils.no_dispatch():
                meta_in_tls = torch._C._meta_in_tls_dispatch_include()
                torch._C._set_meta_in_tls_dispatch_include(True)
                try:
                    func(*args, **kwargs)
                finally:
                    torch._C._set_meta_in_tls_dispatch_include(meta_in_tls)
    if not any((get_write_alias(r) is not None for r in schema_info.outs)):
        return out
    if not all((get_write_alias(r) is not None for r in schema_info.outs)):
        raise RuntimeError('Unsupported schema: ' + str(func._schema))
    if len(func._schema.returns) == 1:
        return get_arg_from_alias(get_write_alias(schema_info.outs[0]), schema_info, args, kwargs)
    outs_to_return = type(out)([get_arg_from_alias(get_write_alias(schema_info.outs[i]), schema_info, args, kwargs) if get_write_alias(r) is not None else o for (i, r), o in zip(enumerate(schema_info.outs), out)])
    return outs_to_return
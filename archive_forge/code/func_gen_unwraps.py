import textwrap
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
from torchgen.api.translate import translate
from torchgen.api.types import DispatcherSignature
from torchgen.context import method_with_native_function
from torchgen.model import (
from torchgen.utils import mapMaybe
def gen_unwraps(flat_arguments: Sequence[Argument], cur_level_var: str) -> Tuple[str, List[str]]:
    arg_names = [a.name for a in flat_arguments]
    arg_types = [a.type for a in flat_arguments]
    tensors = [name for typ, name in zip(arg_types, arg_names) if is_tensor(typ)]
    optional_tensors = [name for typ, name in zip(arg_types, arg_names) if is_optional_tensor(typ)]
    unwraps = []
    for tensor in tensors:
        unwraps += unwrap_tensor(tensor, cur_level_var)
    for opt_tensor in optional_tensors:
        unwraps += unwrap_optional_tensor(opt_tensor, cur_level_var)
    unwrap_code = '\n'.join(unwraps)
    unwrapped_arg_list = []
    for arg in arg_names:
        if arg in tensors or arg in optional_tensors:
            unwrapped_arg_list += [f'{arg}_value', f'{arg}_bdim']
        else:
            unwrapped_arg_list.append(arg)
    return (unwrap_code, unwrapped_arg_list)
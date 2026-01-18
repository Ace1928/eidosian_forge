import textwrap
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
from torchgen.api.translate import translate
from torchgen.api.types import DispatcherSignature
from torchgen.context import method_with_native_function
from torchgen.model import (
from torchgen.utils import mapMaybe
def gen_returns(returns: Tuple[Return, ...], cur_level_var: str, results_var: str) -> str:
    idx = 0
    wrapped_returns = []
    for ret in returns:
        if is_tensor(ret.type):
            wrapped_returns.append(f'makeBatched(std::get<{idx}>({results_var}), std::get<{idx + 1}>({results_var}), {cur_level_var})')
            idx += 2
        elif is_tensor_list(ret.type):
            wrapped_returns.append(f'makeBatchedVector(std::get<{idx}>({results_var}), std::get<{idx + 1}>({results_var}), {cur_level_var})')
            idx += 2
        else:
            wrapped_returns.append(f'std::get<{idx}>({results_var})')
            idx += 1
    if len(wrapped_returns) == 1:
        result = f'return {wrapped_returns[0]};'
    else:
        result = f'return std::make_tuple({', '.join(wrapped_returns)});'
    return result
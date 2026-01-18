from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.dispatcher as dispatcher
from torchgen.api.translate import translate
from torchgen.api.types import Binding, DispatcherSignature, Expr
from torchgen.context import with_native_function
from torchgen.model import (
from torchgen.utils import concatMap
def gather_nonaliased_inner_rets(func: FunctionSchema, out_var: str) -> List[str]:
    aliased_rets = func.aliased_return_names()
    non_aliased_names = []
    is_out_var_a_tuple = len(func.returns) > 1
    for i, r in enumerate(aliased_rets):
        if r is None:
            non_aliased_names.append(f'std::get<{i}>({out_var})' if is_out_var_a_tuple else out_var)
    return non_aliased_names
import re
from collections import defaultdict
from typing import Any, Counter, Dict, List, Match, Optional, Sequence, Set, Tuple
import yaml
from torchgen.api import cpp
from torchgen.api.autograd import (
from torchgen.api.types import (
from torchgen.context import with_native_function
from torchgen.gen import get_grouped_by_view_native_functions, parse_native_yaml
from torchgen.model import (
from torchgen.utils import concatMap, IDENT_REGEX, split_name_params
from torchgen.yaml_utils import YamlLoader
def create_forward_derivative(f: NativeFunction, formula: str, names: Tuple[str, ...]) -> ForwardDerivative:
    var_names = names
    var_types: Optional[Tuple[Type, ...]] = None
    for r in f.func.returns:
        if r.name in var_names:
            if var_types is None:
                var_types = tuple()
            var_types = var_types + (r.type,)
    if var_types is None:
        if var_names == ('result',):
            assert len(f.func.returns) == 1
            var_types = (f.func.returns[0].type,)
        else:
            for var_name in var_names:
                res = re.findall('^result(\\d+)$', var_name)
                if len(res) == 1:
                    if var_types is None:
                        var_types = tuple()
                    arg_idx = int(res[0])
                    var_types = var_types + (f.func.returns[arg_idx].type,)
    assert var_types is not None, 'No matching output for forward derivative definition'
    return ForwardDerivative(formula=formula, var_names=var_names, var_types=var_types, required_inputs_fw_grad=None, required_inputs_primal=None, required_original_self_value=False, is_reusing_outplace_formula=False)
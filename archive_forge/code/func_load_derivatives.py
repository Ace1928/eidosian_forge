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
def load_derivatives(derivatives_yaml_path: str, native_yaml_path: str, tags_yaml_path: str) -> Tuple[Dict[FunctionSchema, Dict[str, DifferentiabilityInfo]], Set[str]]:
    global _GLOBAL_LOAD_DERIVATIVE_CACHE
    key = (derivatives_yaml_path, native_yaml_path)
    if key not in _GLOBAL_LOAD_DERIVATIVE_CACHE:
        with open(derivatives_yaml_path) as f:
            definitions = yaml.load(f, Loader=YamlLoader)
        funcs = parse_native_yaml(native_yaml_path, tags_yaml_path).native_functions
        native_functions_with_view_groups = get_grouped_by_view_native_functions(funcs)
        native_functions_without_view_copies = concatMap(lambda g: [g] if isinstance(g, NativeFunction) else list(g.functions(include_copy=False)), native_functions_with_view_groups)
        view_groups = [g for g in native_functions_with_view_groups if isinstance(g, NativeFunctionsViewGroup)]
        functions_by_signature: Dict[FunctionSchema, List[NativeFunction]] = defaultdict(list)
        functions_by_schema: Dict[str, NativeFunction] = {}
        for function in native_functions_without_view_copies:
            functions_by_signature[function.func.signature()].append(function)
            assert str(function.func) not in functions_by_schema
            functions_by_schema[str(function.func)] = function
        op_counter = Counter[str]()
        infos: Dict[FunctionSchema, Dict[str, DifferentiabilityInfo]] = {}
        used_dispatch_keys: Set[str] = set()
        for defn_dict in definitions:
            if 'dispatch' not in defn_dict:
                specification = defn_dict.pop('name')
                output_differentiability = defn_dict.pop('output_differentiability', None)
                defn_dict = {'name': specification, 'dispatch': {'Default': defn_dict}}
                if output_differentiability:
                    defn_dict['output_differentiability'] = output_differentiability
            name, per_dispatch_diffinfos = create_differentiability_info(defn_dict, functions_by_signature, functions_by_schema, op_counter, used_dispatch_keys)
            infos[name] = per_dispatch_diffinfos
        add_view_copy_derivatives(infos, view_groups)
        _GLOBAL_LOAD_DERIVATIVE_CACHE[key] = (infos, used_dispatch_keys)
    return _GLOBAL_LOAD_DERIVATIVE_CACHE[key]
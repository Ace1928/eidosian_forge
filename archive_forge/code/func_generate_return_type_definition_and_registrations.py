import itertools
import re
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import yaml
from torchgen.api import cpp
from torchgen.api.python import (
from torchgen.code_template import CodeTemplate
from torchgen.context import with_native_function
from torchgen.gen import cpp_string, parse_native_yaml, parse_tags_yaml
from torchgen.model import (
from torchgen.utils import FileManager, split_name_params
from torchgen.yaml_utils import YamlLoader
from .gen_trace_type import should_trace
def generate_return_type_definition_and_registrations(overloads: Sequence[PythonSignatureNativeFunctionPair]) -> Tuple[List[str], List[str]]:
    """
    Generate block of function in `python_return_types.cpp` to initialize
    and return named tuple for a native function which returns named tuple
    and registration invocations in same file.
    """
    typenames: Dict[str, str] = {}
    definitions: List[str] = []
    registrations: List[str] = []
    for overload in overloads:
        fieldnames = namedtuple_fieldnames(overload.function.func.returns)
        if not fieldnames:
            continue
        fields = ', '.join((f'{{"{fn}", ""}}' for fn in fieldnames))
        name = cpp.name(overload.function.func)
        tn_key = gen_namedtuple_typename_key(overload.function)
        typename = typenames.get(tn_key)
        if typename is None:
            typename = f'{name}NamedTuple{('' if not definitions else len(definitions))}'
            typenames[tn_key] = typename
            definitions.append(f'PyTypeObject* get_{name}_namedtuple() {{\n    static PyStructSequence_Field NamedTuple_fields[] = {{ {fields},  {{nullptr}} }};\n    static PyTypeObject {typename};\n    static bool is_initialized = false;\n    static PyStructSequence_Desc desc = {{ "torch.return_types.{name}", nullptr, NamedTuple_fields, {len(fieldnames)} }};\n    if (!is_initialized) {{\n        PyStructSequence_InitType(&{typename}, &desc);\n        {typename}.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;\n        is_initialized = true;\n    }}\n    return &{typename};\n}}\n')
            registrations.append(f'addReturnType(return_types_module, "{name}", generated::get_{name}_namedtuple());')
    return (definitions, registrations)
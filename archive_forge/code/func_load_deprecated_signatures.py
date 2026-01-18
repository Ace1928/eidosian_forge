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
def load_deprecated_signatures(pairs: Sequence[PythonSignatureNativeFunctionPair], deprecated_yaml_path: str, *, method: bool, pyi: bool) -> List[PythonSignatureNativeFunctionPair]:
    grouped: Dict[str, List[PythonSignatureNativeFunctionPair]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.signature.name].append(pair)
    results: List[PythonSignatureNativeFunctionPair] = []
    with open(deprecated_yaml_path) as f:
        deprecated_defs = yaml.load(f, Loader=YamlLoader)
    for deprecated in deprecated_defs:
        schema = FunctionSchema.parse(deprecated['name'])
        aten_name, call_args = split_name_params(deprecated['aten'])
        is_out = aten_name.endswith('_out')
        if is_out:
            aten_name = aten_name.replace('_out', '')
        known_constants = {'1': Type.parse('Scalar')}
        schema_args_by_name = {a.name: a for a in schema.arguments.flat_all}
        for name in call_args:
            assert name in schema_args_by_name or name in known_constants, f'deprecation definiton: Unrecognized value {name}'

        def is_schema_compatible(aten_schema: FunctionSchema) -> bool:
            arguments: Iterable[Argument]
            if is_out:
                arguments = itertools.chain(aten_schema.arguments.out, aten_schema.arguments.flat_non_out)
            else:
                arguments = aten_schema.arguments.flat_all
            for i, arg in enumerate(arguments):
                if i < len(call_args):
                    arg_name = call_args[i]
                    if arg_name in known_constants:
                        schema_type = known_constants[arg_name]
                        schema_annotation = None
                    else:
                        schema_arg = schema_args_by_name[arg_name]
                        schema_type = schema_arg.type
                        schema_annotation = schema_arg.annotation
                    if schema_type != arg.type or schema_annotation != arg.annotation:
                        return False
                elif arg.default is None:
                    return False
            return len(schema.returns) == len(aten_schema.returns) and all((a == b for a, b in zip(schema.returns, aten_schema.returns)))
        any_schema_found = False
        for pair in grouped[aten_name]:
            if not is_schema_compatible(pair.function.func):
                continue
            any_schema_found = True
            python_sig = signature_from_schema(schema, category_override=pair.function.category_override, method=method, pyi=pyi)
            results.append(PythonSignatureNativeFunctionPair(signature=PythonSignatureDeprecated(name=python_sig.name, input_args=python_sig.input_args, input_kwargs=python_sig.input_kwargs, output_args=python_sig.output_args, tensor_options_args=python_sig.tensor_options_args, method=python_sig.method, deprecated_schema=schema, deprecated_args_exprs=tuple(call_args), returns=python_sig.returns), function=pair.function))
        assert any_schema_found, f'No native function with name {aten_name} matched signature:\n  {str(schema)}'
    return results